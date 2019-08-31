def create_logger(name="default", DEBUG=False):
    import logging, sys
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.flush = sys.stdout.flush
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
