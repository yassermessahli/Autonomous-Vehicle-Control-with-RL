import os
import subprocess
import sys

# Configuration
SUMO_ENV_DIR = "sumo_env"
NET_FILE = "highway.net.xml"
ROU_FILE = "highway.rou.xml"
CFG_FILE = "highway.sumocfg"
NOD_FILE = "highway.nod.xml"
EDG_FILE = "highway.edg.xml"


def create_sumo_files():
    if not os.path.exists(SUMO_ENV_DIR):
        os.makedirs(SUMO_ENV_DIR)

    # 1. Create Node File
    with open(os.path.join(SUMO_ENV_DIR, NOD_FILE), "w") as f:
        f.write(
            """<nodes>
    <node id="n0" x="0.0" y="0.0" />
    <node id="n1" x="1000.0" y="0.0" />
</nodes>
"""
        )

    # 2. Create Edge File (2 lanes, 1000m long)
    with open(os.path.join(SUMO_ENV_DIR, EDG_FILE), "w") as f:
        f.write(
            """<edges>
    <edge id="E0" from="n0" to="n1" priority="1" numLanes="2" speed="30.0" />
</edges>
"""
        )

    # 3. Generate Network File using netconvert
    # Assuming netconvert is in PATH. If not, user needs to add it.
    print("Generating network file...")
    try:
        subprocess.run(
            [
                "netconvert",
                "--node-files",
                os.path.join(SUMO_ENV_DIR, NOD_FILE),
                "--edge-files",
                os.path.join(SUMO_ENV_DIR, EDG_FILE),
                "-o",
                os.path.join(SUMO_ENV_DIR, NET_FILE),
            ],
            check=True,
        )
        print(f"Successfully created {NET_FILE}")
    except FileNotFoundError:
        print("Error: 'netconvert' command not found. Please ensure SUMO is installed and added to PATH.")
        print("You can manually generate the net file using the created .nod.xml and .edg.xml files.")
    except subprocess.CalledProcessError as e:
        print(f"Error running netconvert: {e}")

    # 4. Create Route File
    # We define types for ego and obstacle.
    # Actual vehicles will be added dynamically by the Gym Env, but we need the types here.
    with open(os.path.join(SUMO_ENV_DIR, ROU_FILE), "w") as f:
        f.write(
            """<routes>
    <vType id="ego" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="30" color="0,1,0"/>
    <vType id="obstacle" accel="0.0" decel="0.0" sigma="0.0" length="5" minGap="2.5" maxSpeed="0" color="1,0,0"/>
    
    <route id="r0" edges="E0"/>
</routes>
"""
        )

    # 5. Create SUMO Config File
    with open(os.path.join(SUMO_ENV_DIR, CFG_FILE), "w") as f:
        f.write(
            f"""<configuration>
    <input>
        <net-file value="{NET_FILE}"/>
        <route-files value="{ROU_FILE}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="10000"/>
    </time>
    <report>
        <verbose value="true"/>
        <no-step-log value="true"/>
    </report>
</configuration>
"""
        )
    print(f"Successfully created {CFG_FILE}")


if __name__ == "__main__":
    create_sumo_files()
