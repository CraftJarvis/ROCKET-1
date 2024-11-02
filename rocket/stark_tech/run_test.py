from rocket.stark_tech.entry import env_generator


def main():
    env, _ = env_generator()

    obs = env.reset()

    action = env.noop_action()
    action["voxels"] = [-3, 3, -3, 3, -3, 3]
    action["mobs"] = [-200, 200, -200, 200, -200, 200]

    obs, r, d, info = env.step(action)

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()