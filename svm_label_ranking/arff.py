# -*- coding: utf-8 -*-
# Copyright (c) 2008, Mikio L. Braun, Cheng Soon Ong, Soeren Sonnenburg
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#     * Neither the names of the Technical University of Berlin, ETH
# Zürich, or Fraunhofer FIRST nor the names of its contributors may be
# used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import re
import numpy as np
import copy


class ArffFile(object):
    """class to read and write arff data structures

    An ARFF File object describes a data set consisting of a number
    of data points made up of attributes. The whole data set is called
    a 'relation'. Supported attributes are:

    - 'numeric': floating point numbers
    - 'string': strings
    - 'nominal': taking one of a number of possible values
    - 'ranking': taking a ranking of labels (using the xarff format of WEKA-LR)

    Not all features of ARFF files are supported yet. The most notable
    exceptions are:

    - no sparse data
    - no support for date and relational attributes

    Also, parsing of strings might still be a bit brittle.

    You can either load or save from files, or write and parse from a
    string.

    You can also construct an empty ARFF file and then fill in your
    data by hand. To define attributes use the define_attribute method.

    Attributes are:

    :param relation: name of the relation
    :type relation: string
    :param attributes: names of attributes
    :type attributes: list of strings
    :param attribute_types: types of attributes (dictionnary keys: names of att.)
    :type attribute_types: dictionnary
    :param attribute_data: data of attributes (modalities for non-numeric)
    :type attribute_data: None or list of strings
    :param comment: the initial comment in the file. Typically contains some
        information on the data set.
    :type comment: string
    :param data: the actual data of the file
    :type data: list of items

    .. todo::

        * transform dump into __str__ method

    """

    def __init__(self):
        """Construct an empty ARFF structure."""
        self.relation = ""
        self.attributes = []
        self.attribute_types = dict()
        self.attribute_data = dict()
        self.comment = []
        self.data = []
        pass

    def load(self, filename):
        """Load an ARFF File from a file.

        :param filename: the name of the file containing data
        :type filename: string
        """

        # reinitialize the object
        self.__init__()
        # fill in the object
        o = open(filename)
        s = o.read()
        a = ArffFile.parse(s)
        self.relation = a.relation
        self.attributes = a.attributes
        self.attribute_types = a.attribute_types
        self.attribute_data = a.attribute_data
        self.comment = a.comment
        self.data = a.data
        o.close()

    def select_class(self, select):
        """return an ARFF object where only some classes are selected

        :param select: the names of the classes to retain
        :type select: list
        :return: a new ArffFile structure containing only selected classes
        :rtype: :class:`arff.ArffFile`

        .. warning::

            should not be used with files containing ranks
        """
        if "class" not in self.attribute_data.keys():
            raise NameError("Cannot find a class attribute.")
        if set(select) - set(self.attribute_data["class"]) != set([]):
            raise NameError("Specified classes not a subset of existing ones!")
        selection = ArffFile()
        # construct list with sets of indices matching class names
        # assume the class is the last provided item
        indices = [i for i, val in enumerate(self.data) if val[-1] in select]
        # data corresponding to provided class
        selected_data = [self.data[i] for i in indices]

        selection.attribute_data = self.attribute_data.copy()
        selection.attribute_types = self.attribute_types.copy()
        selection.attribute_data["class"] = select
        selection.data = selected_data
        selection.relation = self.relation
        selection.attributes = self.attributes[:]
        selection.comment = self.comment[:]

        return selection

    def select_col_vals(self, column, select):
        """return an ARFF File where only some rows are selected in data

        :param select: the values to retain
        :type select: list
        :param column: name of the attribute
        :type column: string
        :return: a new ArffFile structure containing only selected values in the column
        :rtype: :class:`arff.ArffFile`
        """

        if column not in self.attribute_data.keys():
            raise NameError("Cannot find specified column.")
        selection = ArffFile()
        col_ind = self.attributes.index(column)
        # construct list with sets of indices matching infos
        indices = [i for i, val in enumerate(self.data) if val[col_ind] in select]

        selected_data = [self.data[i] for i in indices]

        selection.attribute_data = self.attribute_data.copy()
        selection.attribute_types = self.attribute_types.copy()
        if self.attribute_types[column] == "nominal":
            selection.attribute_data[column] = select
        selection.data = selected_data
        selection.relation = self.relation
        selection.attributes = self.attributes[:]
        selection.comment = self.comment[:]

        return selection

    def select_class_binary(self, positive, negative):
        """return an ARFF object where only some classes are selected in order
        to form a dataset for a binary classification problem.

        :param select: the names of the classes to retain
        :type select: list

        :param positive: classes values to be considered as 'positive' in the binary
        classification problem
        :type positive:list

        :param positive: classes values to be considered as 'positive' in the binary
        classification problem
        :type positive:list

        :returns: a new ArffFile structure containing only selected classes
        :rtype: list of strings, :class:`arff.ArffFile`


        """
        if "class" not in self.attribute_data.keys():
            raise NameError("Cannot find a class attribute.")
        if set(positive) - set(self.attribute_data["class"]) != set([]):
            raise NameError(
                "Specified 'positive' classes not a subset of existing ones!"
            )

        if set(negative) - set(self.attribute_data["class"]) != set([]):
            raise NameError(
                "Specified 'negative' classes not a subset of existing ones!"
            )

        selection = copy.deepcopy(self)
        # initiate the variable for the output data
        selected_data = []

        # construct list with sets of indices matching class names
        # assume the class is the last provided item
        # and form data corresponding to provided class
        for i, val in enumerate(self.data):

            if val[-1] in positive:
                buff = self.data[i][:]
                buff[-1] = "positive"
                selected_data.append(buff)

            elif val[-1] in negative:
                buff = self.data[i][:]
                buff[-1] = "negative"
                selected_data.append(buff)

        selection.attribute_data["class"] = ["positive", "negative"]
        selection.data = selected_data

        return selection

    def remove_col(self, column):
        """return an ARFF File where the specified column is removed

        :param column: name of the attribute
        :type column: string
        :return: a new ArffFile structure excluding the specified column
        :rtype: :class:`arff.ArffFile`
        """
        if column not in self.attribute_data.keys():
            raise NameError("Cannot find specified column.")

        selection = self
        col_ind = self.attributes.index(column)
        del selection.attributes[col_ind]
        del selection.attribute_types[column]
        del selection.attribute_data[column]
        selection.data = [row[0:col_ind] + row[(col_ind + 1) :] for row in self.data]

        return selection

    def make_clone(self):
        """Make a copy of the current object

        :return: a copy
        :rtype: :class:`arff.ArffFile`
        """
        cloned = ArffFile()

        cloned.attribute_data = self.attribute_data.copy()
        cloned.attribute_types = self.attribute_types.copy()
        cloned.data = copy.deepcopy(self.data)
        cloned.relation = self.relation
        cloned.attributes = copy.deepcopy(self.attributes)
        cloned.comment = copy.deepcopy(self.comment)

        return cloned

    def discretize(self, discmet, numint=4, selfeat=None):
        """Discretize selected features (if none, then discretize all numeric
        ones) according to specified method and number of intervals.

        if discmet='eqfreq', discretization is done so that each interval have
        equal frequencies in the data set

        if discmet='eqwidth', discretization is done so that each interval has
        equal length, according to maximum and minimum of data set

        if discmet='ent', the method of [#fayyad1993]_

        :param discmet: discretization method. Can be either 'ent','eqfreq' or 'eqwidth'
        :type discmet: string
        :param numint: number of intervals into which divide attributes
        :type numint: integer
        :param selfeat: name of a particular feature to discretize, if None discretize all
        :type selfeat: string

        ..todo::
            * encode the method of fayyad et al. 1993 in this function (rather than using Orange)
        """
        datasave = np.array([map(str, x) for x in self.data]).astype("|S25")
        numitem = datasave.shape[0]

        if discmet == "eqfreq":
            if selfeat != None:
                if self.attribute_types[selfeat] != "numeric":
                    raise NameError("Selected feature not numeric.")
                indexfeat = self.attributes.index(selfeat)
                floatdata = datasave[:, indexfeat].astype(float)
                datasave = datasave[np.argsort(floatdata)]
                floatdata = floatdata[np.argsort(floatdata)]
                cutpoint = []
                newname = []
                for i in range(numint):
                    cutpoint.append(
                        datasave[((i + 1) * (numitem / (numint))) - 1, indexfeat]
                    )
                for i in range(numint):
                    if i == 0:
                        string = str(cutpoint[i])
                        newname.append("<=" + string[0 : min(len(string), 7)])
                    elif i == (numint - 1):
                        string = str(cutpoint[i - 1])
                        newname.append(">" + string[0 : min(len(string), 7)])
                    else:
                        string1 = str(cutpoint[i - 1])
                        string2 = str(cutpoint[i])
                        newname.append(
                            "("
                            + string1[0 : min(len(string1), 7)]
                            + ";"
                            + string2[0 : min(len(string2), 7)]
                            + "]"
                        )
                for i in range(numint):
                    if i == 0:
                        datasave[
                            (floatdata <= cutpoint[i].astype(float)), indexfeat
                        ] = newname[i]
                    elif i == (numint - 1):
                        datasave[
                            (floatdata > cutpoint[i - 1].astype(float)), indexfeat
                        ] = newname[i]
                    else:
                        datasave[
                            (floatdata > cutpoint[i - 1].astype(float))
                            & (floatdata <= cutpoint[i].astype(float)),
                            indexfeat,
                        ] = newname[i]
                self.data = datasave.tolist()
                self.attribute_types[selfeat] = "nominal"
                self.attribute_data[selfeat] = newname
            else:
                for i in range(len(self.attributes)):
                    feature = self.attributes[i]
                    if self.attribute_types[feature] == "numeric":
                        floatdata = datasave[:, i].astype(float)
                        datasave = datasave[np.argsort(floatdata)]
                        floatdata = floatdata[np.argsort(floatdata)]
                        cutpoint = []
                        newname = []
                        for j in range(numint):
                            cutpoint.append(
                                datasave[((j + 1) * (numitem / (numint))) - 1, i]
                            )
                        for j in range(numint):
                            if j == 0:
                                string = str(cutpoint[j])
                                newname.append("<=" + string[0 : min(len(string), 7)])
                            elif j == (numint - 1):
                                string = str(cutpoint[j - 1])
                                newname.append(">" + string[0 : min(len(string), 7)])
                            else:
                                string1 = str(cutpoint[j - 1])
                                string2 = str(cutpoint[j])
                                newname.append(
                                    "("
                                    + string1[0 : min(len(string1), 7)]
                                    + ";"
                                    + string2[0 : min(len(string2), 7)]
                                    + "]"
                                )
                        for j in range(numint):
                            if j == 0:
                                datasave[
                                    (floatdata <= cutpoint[j].astype(float)), i
                                ] = newname[j]
                            elif j == (numint - 1):
                                datasave[
                                    (floatdata > cutpoint[j - 1].astype(float)), i
                                ] = newname[j]
                            else:
                                datasave[
                                    (floatdata > cutpoint[j - 1].astype(float))
                                    & (floatdata <= cutpoint[j].astype(float)),
                                    i,
                                ] = newname[j]
                        self.attribute_types[feature] = "nominal"
                        self.attribute_data[feature] = newname
                self.data = datasave.tolist()

        if discmet == "eqwidth":
            if selfeat != None:
                if self.attribute_types[selfeat] != "numeric":
                    raise NameError("Selected feature not numeric.")
                indexfeat = self.attributes.index(selfeat)
                floatdata = datasave[:, indexfeat].astype(float)
                cutpoint = []
                newname = []
                totalwidth = floatdata.max() - floatdata.min()
                for i in range(numint):
                    cutpoint.append(floatdata.min() + (i + 1) * totalwidth / numint)
                for i in range(numint):
                    if i == 0:
                        string = str(cutpoint[i])
                        newname.append("<=" + string[0 : min(len(string), 7)])
                    elif i == (numint - 1):
                        string = str(cutpoint[i - 1])
                        newname.append(">" + string[0 : min(len(string), 7)])
                    else:
                        string1 = str(cutpoint[i - 1])
                        string2 = str(cutpoint[i])
                        newname.append(
                            "("
                            + string1[0 : min(len(string1), 7)]
                            + ";"
                            + string2[0 : min(len(string2), 7)]
                            + "]"
                        )
                for i in range(numint):
                    if i == 0:
                        datasave[(floatdata <= cutpoint[i]), indexfeat] = newname[i]
                    elif i == (numint - 1):
                        datasave[(floatdata > cutpoint[i - 1]), indexfeat] = newname[i]
                    else:
                        datasave[
                            (floatdata > cutpoint[i - 1]) & (floatdata <= cutpoint[i]),
                            indexfeat,
                        ] = newname[i]
                self.data = datasave.tolist()
                self.attribute_types[selfeat] = "nominal"
                self.attribute_data[selfeat] = newname
            else:
                for i in range(len(self.attributes)):
                    feature = self.attributes[i]
                    if self.attribute_types[feature] == "numeric":
                        floatdata = datasave[:, i].astype(float)
                        cutpoint = []
                        newname = []
                        totalwidth = floatdata.max() - floatdata.min()
                        for j in range(numint):
                            cutpoint.append(
                                floatdata.min() + (j + 1) * totalwidth / numint
                            )
                        for j in range(numint):
                            if j == 0:
                                string = str(cutpoint[j])
                                newname.append("<=" + string[0 : min(len(string), 7)])
                            elif j == (numint - 1):
                                string = str(cutpoint[j - 1])
                                newname.append(">" + string[0 : min(len(string), 7)])
                            else:
                                string1 = str(cutpoint[j - 1])
                                string2 = str(cutpoint[j])
                                newname.append(
                                    "("
                                    + string1[0 : min(len(string1), 7)]
                                    + ";"
                                    + string2[0 : min(len(string2), 7)]
                                    + "]"
                                )
                        for j in range(numint):
                            if j == 0:
                                datasave[(floatdata <= cutpoint[j]), i] = newname[j]
                            elif j == (numint - 1):
                                datasave[(floatdata > cutpoint[j - 1]), i] = newname[j]
                            else:
                                datasave[
                                    (floatdata > cutpoint[j - 1])
                                    & (floatdata <= cutpoint[j]),
                                    i,
                                ] = newname[j]
                        self.attribute_types[feature] = "nominal"
                        self.attribute_data[feature] = newname
                self.data = datasave.tolist()

        if discmet == "ent":
            print("sorry, not implemented yet")

    @staticmethod
    def parse(s):
        """Parse an ARFF File already loaded into a string."""
        a = ArffFile()
        a.state = "comment"
        a.lineno = 1
        for l in s.splitlines():
            a.__parseline(l)
            # print(a.lineno)
            a.lineno += 1
        return a

    def save(self, filename):
        """Save an arff structure to a file.

        :param filename: the name of the file where data are saved
        :type filename: string

        """
        o = open(filename, "w")
        o.write(self.write())
        o.close()

    def write(self):
        """Write an arff structure to a string."""
        o = []
        o.append("% " + re.sub("\n", "\n% ", self.comment))
        o.append("@relation " + self.esc(self.relation))
        for a in self.attributes:
            at = self.attribute_types[a]
            if at == "numeric":
                o.append("@attribute " + self.esc(a) + " numeric")
            elif at == "string":
                o.append("@attribute " + self.esc(a) + " string")
            elif at == 'nominal':
                o.append("@attribute " + self.esc(a) +
                         " {" + ','.join(self.attribute_data[a]) + "}")
            elif at == 'ranking':
                o.append("@attribute" + self.esc(a) + " ranking" +
                         " {" + ','.join(self.attribute_data[a]) + "}")
            else:
                raise NameError("Type " + at + " not supported for writing!")
        o.append("\n@data")
        for d in self.data:
            line = []
            for e, a in zip(d, self.attributes):
                at = self.attribute_types[a]
                if at == "numeric":
                    line.append(str(e))
                elif at == "string":
                    line.append(self.esc(e))
                elif at == "nominal":
                    line.append(e)
                elif at == "ranking":
                    line.append(e)
                else:
                    raise "Type " + at + " not supported for writing!"
            o.append(",".join(line))
        return "\n".join(o) + "\n"

    def esc(self, s):
        "Escape a string if it contains spaces"
        if re.match(r"\s", s):
            return "'" + s + "'"
        else:
            return s

    def define_attribute(self, name, atype, data=None):
        """Define a new attribute.

        For nominal and ranking attributes, pass the possible values as data.

        :param atype: 'numeric', 'string', 'ranking' and 'nominal'.
        :type atype: string
        :param name: name of the attribute
        :type name: string
        :param data: modalities/labels of the attribute
        :type atype: list
        """

        self.attributes.append(name)
        self.attribute_types[name] = atype
        self.attribute_data[name] = data

    def get_features_matrix(self):
        """
        :return: a numpy matrix of features
        :rtype: :class:`numpy.array`
        """
        col_ind = self.attributes.index("L")
        features_list = [row[0:col_ind] + row[(col_ind + 1) :] for row in self.data]
        return np.array(features_list)

    def __parseline(self, l):
        if self.state == "comment":
            if len(l) > 0 and l[0] == "%":
                self.comment.append(l[2:])
            else:
                self.comment = "\n".join(self.comment)
                self.state = "in_header"
                self.__parseline(l)
        elif self.state == "in_header":
            ll = l.lower()
            if ll.startswith("@relation "):
                self.__parse_relation(l)
            if ll.startswith("@attribute "):
                self.__parse_attribute(l)
            if ll.startswith("@data"):
                self.state = "data"
        elif self.state == "data":
            if len(l) > 0 and l[0] == "{":
                self.__parse_sparse_data(l)
            elif len(l) > 0 and l[0] != "%":
                self.__parse_data(l)

    def __parse_relation(self, l):
        l = l.split()
        self.relation = l[1]

    def __parse_attribute(self, l):
        p = re.compile(r"[a-zA-Z_-][a-zA-Z0-9_-]*|\{[^\}]+\}|\'[^\']+\'|\"[^\"]+\"")
        l = [s.strip() for s in p.findall(l)]
        name = l[1].replace("'", "")  # Modification : all ' in typename are removed
        atype = l[2]
        atypel = atype.lower()
        if atypel == "real" or atypel == "numeric" or atypel == "integer":
            self.define_attribute(name, "numeric")
        elif atypel == "string":
            self.define_attribute(name, "string")
        elif atypel == "ranking":
            labelrow = l[3]
            labels = [s.strip() for s in labelrow[1:-1].split(",")]
            self.define_attribute(name, "ranking", labels)
        elif atype[0] == "{" and atype[-1] == "}":
            values = [s.strip() for s in atype[1:-1].split(",")]
            self.define_attribute(name, "nominal", values)
        else:
            self.__print_warning(
                "unsupported type " + atype + " for attribute " + name + "."
            )

    def __append_data(self, datum, attrib, value):
        at = self.attribute_types[attrib]
        if at == "numeric":
            if re.match(r"[+-]?[0-9]+(?:\.[0-9]*(?:[eE]-?[0-9]+)?)?", value):
                datum.append(float(value))
            elif value == "?":
                datum.append(float("nan"))
            else:
                self.__print_warning(
                    "non-numeric value %s for numeric attribute %s"
                    % (value, self.attributes[idx_attrib])
                )
                return
        elif at == "string":
            datum.append(value)
        elif at == "ranking":
            for k in value.split(">"):
                if k not in self.attribute_data[attrib]:
                    self.__print_warning(
                        "incorrect label %s for ranking attribute %s"
                        % (value, self.attributes[idx_attrib])
                    )
            datum.append(value)
        elif at == "nominal":
            if value in self.attribute_data[attrib]:
                datum.append(value)
            elif value == "?":
                datum.append(None)
            else:
                self.__print_warning(
                    "incorrect value %s for nominal attribute %s"
                    % (value, self.attributes[idx_attrib])
                )

    def __parse_sparse_data(self, l):
        pairs = [s.strip() for s in l.strip("{}").split(",")]
        nonzero = []
        for u in pairs:
            nonzero.append(tuple(u.split(" ")))
        nonzero = sorted(nonzero, key=lambda x: int(x[0]))
        i = 0
        datum = []
        for j in range(0, len(self.attributes)):
            if i < len(nonzero) and int(nonzero[i][0]) == j:
                self.__append_data(datum, self.attributes[j], nonzero[i][1])
                i = i + 1
            else:
                self.__append_data(datum, self.attributes[j], "0")
        self.data.append(datum)

    def __parse_data(self, l):
        l = [s.strip() for s in l.split(",")]
        if len(l) != len(self.attributes):
            return

        datum = []
        for n, v in zip(self.attributes, l):
            self.__append_data(datum, n, v)
        self.data.append(datum)

    def __print_warning(self, msg):
        print(("Warning (line %d): " % self.lineno) + msg)

    def dump(self):
        """Print an overview of the ARFF file."""
        print("Relation " + self.relation)
        print("  With attributes")
        for n in self.attributes:
            if self.attribute_types[n] != "nominal":
                print("    %s of type %s" % (n, self.attribute_types[n]))
            else:
                print(
                    "    "
                    + n
                    + " of type nominal with values "
                    + ", ".join(self.attribute_data[n])
                )
        for d in self.data:
            print(d)
