from setuptools import find_packages, setup

package_name = 'visual_pubsub'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='carlos',
    maintainer_email='menachocarlos5@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sub_joints = visual_pubsub.sub_joints:main',
            'pub_joints = visual_pubsub.pub_joints:main',
            'inverse_k = visual_pubsub.inverse_kinematics:main',
            'inverse_k_p = visual_pubsub.inverse_kinematics_pulgar:main',
        ],
    },
)