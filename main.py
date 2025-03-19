# !/usr/bin/python3
from mr_playground.engine import IRSim



def main():
    engine = IRSim()
    engine.init("","")

    while engine.ok():
        engine.step()
        engine.render()
        if engine._env.done():
            break
    engine._env.end()


if __name__ == "__main__":
    main()
