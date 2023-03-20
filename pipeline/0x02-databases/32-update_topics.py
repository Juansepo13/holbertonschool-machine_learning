#!/usr/bin/env python3
"""
Function that changes school topics
"""


def update_topics(mongo_collection, name, topics):
    """
    Changes school topics
    """
    search = {"name": name}
    new = {"$set": {"topics": topics}}

    mongo_collection.update_many(search, new)
