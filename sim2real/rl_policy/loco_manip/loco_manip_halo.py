import sys

import argparse
import yaml

sys.path.append("../")
sys.path.append("./rl_policy")

from sim2real.rl_policy.loco_manip.loco_manip import LocoManipPolicy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof_halo.yaml", help="config file")
    parser.add_argument("--model_path", type=str, help="path to the ONNX model file")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    model_path = args.model_path if args.model_path else config.get("model_path")
    if not model_path:
        raise ValueError("model_path must be provided either via --model_path argument or in config file")

    policy = LocoManipPolicy(
        config=config, model_path=model_path, rl_rate=50, policy_action_scale=0.25
    )
    policy.run()
