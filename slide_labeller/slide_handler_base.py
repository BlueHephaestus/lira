"""
All our useful extraneous functions for our scripts to make use of

-Blake Edwards / Dark Element
"""

import json
def archive_metadata(metadata, metadata_dir):
    """
    Takes as input a metadata dictionary, and stores it in our metadata_dir txt file as a json string.
    """
    with open(metadata_dir, "w") as f:
        json.dumps(metadata, f)


