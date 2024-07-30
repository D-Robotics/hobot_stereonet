from setuptools import setup

package_name = 'hobot_stereonet_render'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kao.zhu',
    maintainer_email='kao.zhu@d-robotics.cc',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],

    entry_points={
        'console_scripts': [
                'talker = hobot_stereonet_render.publisher_member_function:main',
        ],
    },


)
