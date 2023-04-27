import os


#
# General utilities
#
def create_directories(file_path, mode=0o755):
    """Create directories and subdirectories for a file_path 
    if they do not exist. Doesn't create the file though.
    Usage Example: 
        create_directories('/home/huej/layer1/layer2/test.json')
    
    Parameters
    ----------
    file_path: str
    mode: octal number default: 0o755 == rwxr-xr-x user-group-other permissions
    """
    # Check if the file path exists
    if not os.path.exists(os.path.dirname(file_path)):
    # Create the necessary directories if they don't exist
        os.makedirs(os.path.dirname(file_path), mode=mode)