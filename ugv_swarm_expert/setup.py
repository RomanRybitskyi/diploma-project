from glob import glob

from setuptools import find_packages, setup

package_name = "ugv_swarm_expert"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/worlds", glob("worlds/*.sdf")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
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
            "expert_data_collector = ugv_swarm_expert.data.expert_data_collector:main",
            "dataset_preprocessor = ugv_swarm_expert.data.dataset_preprocessor:main",
            "feature_engineer = ugv_swarm_expert.data.feature_engineer:main",
            "inference_node = ugv_swarm_expert.inference.inference_node:main",
            "eval_runner = ugv_swarm_expert.evaluation.eval_metrics:main",
            "ma_gail_train = ugv_swarm_expert.training.train:main",
            "leader_navigator = ugv_swarm_expert.navigation.leader_navigator:main",
        ],
    },
)
