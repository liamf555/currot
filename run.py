import argparse
from deep_sprl.util.parameter_parser import parse_parameters
import deep_sprl.environments
import torch


def main():
    parser = argparse.ArgumentParser("Self-Paced Learning experiment runner")
    parser.add_argument("--base_log_dir", type=str, default="logs")
    parser.add_argument("--type", type=str, default="wasserstein",
                        choices=["default", "random", "self_paced", "wasserstein", "alp_gmm",
                                 "goal_gan", "acl", "plr", "vds"])
    parser.add_argument("--learner", type=str, default="ppo", choices=["ppo", "sac", "droq"])
    parser.add_argument("--env", type=str, default="point_mass_2d", choices=["point_mass_2d", "gym_mxs/MXSBox2D-v0", "gym_mxs/MXSBox2DLidar-v0"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_cores", type=int, default=1)
    parser.add_argument("--config_dir", type=str, default="./config/box_env_acl_config.json")

    args, remainder = parser.parse_known_args()
    parameters = parse_parameters(remainder)

    torch.set_num_threads(args.n_cores)

    if args.env == "point_mass_2d":
        from deep_sprl.experiments import PointMass2DExperiment
        exp = PointMass2DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)
    elif args.env == "gym_mxs/MXSBox2D-v0" or args.env == "gym_mxs/MXSBox2DLidar-v0":
        from deep_sprl.experiments import MXSBox2DExperiment
        exp = MXSBox2DExperiment(args.base_log_dir, args.type, args.learner, args.env, args.config_dir,  parameters, args.seed)
    else:
        raise RuntimeError("Unknown environment '%s'!" % args.env)

    exp.train()
    # exp.evaluate()
    # shutil.copy(config_file, f"{run_dir}/box_env_config.json")


if __name__ == "__main__":
    main()
