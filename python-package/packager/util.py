"""
Utility functions for implementing PEP 517 backend
"""
import shutil


def copytree_with_logging(src, dest, *, logger):
    """Call shutil.copytree() with logging"""
    logger.info("Copying %s -> %s", str(src), str(dest))
    shutil.copytree(src, dest)


def copy_with_logging(src, dest, *, logger):
    """Call shutil.copy() with logging"""
    if dest.is_dir():
        logger.info("Copying %s -> %s", str(src), str(dest / src.name))
    else:
        logger.info("Copying %s -> %s", str(src), str(dest))
    shutil.copy(src, dest)
