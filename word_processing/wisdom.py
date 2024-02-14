class RandomWisdom:
    wise_sayings = [
        "📚📚📚 Learn to 🧑‍🔬 READ 📚📚📚",
        "Your 🧠 is a muscle. 💪 The more you use it, the stronger it gets. 🏋️‍♂️",
        "📖 Reading is a great way to exercise your brain. 🧠",
        "📖 Reading 📖  It helps you learn new things and understand the world around you. 🌍",
    ]

    def __init__(self):
        import random
        self.wisdom = self.wise_sayings[random.randint(0, len(self.wise_sayings) - 1)]

    def __str__(self):
        return self.wisdom
