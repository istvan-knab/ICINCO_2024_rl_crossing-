import neptune
class Logger(object):
    def __init__(self, config: dict):
        device = config['DEVICE']
        algorithm = config['algorithm']
        environment = config['environment']
        RENDER_MODE = config['RENDER_MODE']
        EPISODES = config['EPISODES']
        SEED = config['SEED']
        ALPHA = config['ALPHA']
        GAMMA = config['GAMMA']
        EPSILON = config['EPSILON']
        EPSILON_DECAY = config['EPSILON_DECAY']

        print("Start training-----------------------------------------------")
        print(f"Device : {device}")
        print(f"Algorithm : {algorithm}")
        print(f"Environment : {environment}")
        print(f"RENDER_MODE : {RENDER_MODE}")
        print(f"EPISODES : {EPISODES}")
        print(f"SEED : {SEED}")
        print(f"ALPHA : {ALPHA}")
        print(f"GAMMA : {GAMMA}")
        print(f"EPSILON : {EPSILON}")
        print(f"EPSILON_DECAY : {EPSILON_DECAY}")
        print("-------------------------------------------------------------")
        self.config = config
        self.run = neptune.init_run(
            project="istvan-knab/reinforcement-learning",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ZGUzMWI0ZC01ZDIyLTQwNWQtODQzOS1mNzQ5NTA3YzdmOGUifQ==",
        )  # your credentials

        params = {"learning_rate": self.config["ALPHA"],
                  "discount factor": self.config["GAMMA"],
                  "optimizer": "Adam"}
        self.run["parameters"] = params


    def step(self,*args) -> None:
        """
        Identification of needed values for logging purposes
        Calling the log function to console and online neptune logger
        :param args: [episode, reward, config, loss]
        :return: None
        """
        self.episode = args[0]
        self.reward = int(args[1])
        self.epsilon = args[2]["EPSILON"]
        self.config = args[2]
        self.loss = args[3]

        self.console()
        self.neptune()

    def neptune(self) -> None:
        """
        Neptune callback, visual evaluation ov learning process
        :return: None
        """
        self.run["train/reward"].append(self.reward)
        self.run["train/epsilon"].append(self.config["EPSILON"])
        self.run["train/loss"].append(self.loss)


    def console(self) -> None:
        """
        Required parameters for learning evaluation on the console
        :return:
        """
        print("================================================")
        print(f"Episode {self.episode}")
        print(f"Reward {self.reward}")
        print(f"Epsilon {self.epsilon}")
        print("================================================")