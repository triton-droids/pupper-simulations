'''
Helper script to test whether the environment builds or not
'''

from bittle_env import BittleEnv

from brax import envs

envs.register_environment('bittle', BittleEnv)

env_name = 'bittle'
xml_path = 'bittle_adapted_scene.xml'

try:
    env = envs.get_environment(env_name, xml_path = xml_path)
    print(f"Successfully built environment {env_name} from {xml_path}")
except Exception as e:
    print(f"Failed to build environment {env_name} from {xml_path}")
    print(e)