
from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'panda_moveit'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),

    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml') + glob('config/*.srdf')
        ),
        (
            os.path.join('share', package_name, 'config', 'urdf'),
            glob('config/urdf/*')
        ),
        (
        os.path.join('share', package_name, 'worlds'),
        glob('worlds/*.sdf')
        ),
        ],

    install_requires=['setuptools'],
    zip_safe=True,

    maintainer='syedrizvi',
    maintainer_email='swh2rizv@uwaterloo.ca',

    description='MoveIt 2 integration for Franka Panda in Gazebo Harmonic',
    license='Apache-2.0',

    extras_require={
        'test': ['pytest'],
    },

    entry_points={
        'console_scripts': [
            "panda_controller = panda_moveit.panda_controller:main",
            "dual_joint_states = panda_moveit.dual_joint_states:main",
            "dual_trajectory_adapter = panda_moveit.dual_trajectory_adapter:main",
            "planning_scene_setup = panda_moveit.planning_scene_setup:main",
            "pick_place = panda_moveit.pick_place:main",
        ],
    },
)
