from setuptools import find_packages, setup

package_name = "ugv_swarm_expert"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "pandas", "torch"],
    zip_safe=True,
    maintainer="Roman",
    maintainer_email="roman@example.com",
    description="Expert demonstration data collection nodes for UGV swarm formation control.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "expert_data_collector = ugv_swarm_expert.expert_data_collector:main",
            "dataset_preprocessor = ugv_swarm_expert.dataset_preprocessor:main",
            "feature_engineer = ugv_swarm_expert.feature_engineer:main",
            "inference_node = ugv_swarm_expert.inference_node:main",
        ],
    },
)
