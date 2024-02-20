import rospkg
import yaml
import os
#########################
#     Configuration     #
#########################

# Returns configuration file bb_filters in package
def get_config(name, package='bb_filters', folder=None):
    rospack = rospkg.RosPack()
    if folder is not None:
        filepath = os.path.join(rospack.get_path(package), 'config', folder, name)
    else:
        filepath = os.path.join(rospack.get_path(package), 'config', name)
    with open(filepath, 'r') as stream:
        return yaml.safe_load(stream)

# Returns list of configuration files
def get_config_files(package='bb_filters', folder=None):
    rospack = rospkg.RosPack()
    if folder is not None:
        filepath = os.path.join(rospack.get_path(package), 'config', folder)
    else:
        filepath = os.path.join(rospack.get_path(package), 'config')
    return [name for name in os.listdir(filepath) if name.endswith('.yaml')]