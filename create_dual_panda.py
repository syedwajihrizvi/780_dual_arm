
import copy
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import xacro

SOURCE = Path("config/urdf/panda.urdf.xacro")
OUTPUT = Path("config/urdf/dual_panda.urdf")

# Expand our existing single-Panda Xacro.
single = ET.fromstring(
    xacro.process_file(str(SOURCE)).toxml()
)

dual = ET.Element("robot", {"name": "dual_panda"})

# Shared virtual root for MoveIt.
ET.SubElement(dual, "link", {"name": "world"})

# Preserve any globally defined materials.
for material in single.findall("material"):
    dual.append(copy.deepcopy(material))

# Each Panda has its own independent kinematic chain.
robots = [
    ("first_", 0.20, 0.0, 1.025, 0.0),
    ("second_", 1.50, 0.0, 1.025, math.pi),
]

for prefix, x, y, z, yaw in robots:

    # Copy and rename every link.
    for link in single.findall("link"):
        new_link = copy.deepcopy(link)
        new_link.set("name", prefix + link.get("name"))
        dual.append(new_link)

    # Copy joints and update all link references.
    for joint in single.findall("joint"):
        new_joint = copy.deepcopy(joint)
        new_joint.set("name", prefix + joint.get("name"))

        for tag in ("parent", "child"):
            element = new_joint.find(tag)
            if element is not None:
                element.set(
                    "link", prefix + element.get("link")
                )

        # Important for the Panda's mimic finger joint.
        mimic = new_joint.find("mimic")
        if mimic is not None:
            mimic.set(
                "joint", prefix + mimic.get("joint")
            )

        dual.append(new_joint)

    # Fixed mounting transform for this Panda.
    # This is a kinematic reference, not a
    # physical joint between the two robots.
    mount = ET.SubElement(
        dual,
        "joint",
        {
            "name": prefix + "world_mount",
            "type": "fixed",
        },
    )

    ET.SubElement(
        mount, "parent", {"link": "world"}
    )

    ET.SubElement(
        mount,
        "child",
        {"link": prefix + "panda_link0"},
    )

    ET.SubElement(
        mount,
        "origin",
        {
            "xyz": f"{x} {y} {z}",
            "rpy": f"0 0 {yaw}",
        },
    )

# xacro's $(find) expands to absolute file:// paths on this machine; use
# package:// so the generated URDF works in any workspace.
for mesh in dual.iter("mesh"):
    mesh.set("filename", re.sub(
        r"^file://.*/share/(moveit_resources_panda_description)/",
        r"package://\1/",
        mesh.get("filename"),
    ))

ET.indent(dual, space="  ")

ET.ElementTree(dual).write(
    OUTPUT,
    encoding="unicode",
    xml_declaration=True,
)

print(f"Created {OUTPUT}")
print("Robot 1: first_panda_*")
print("Robot 2: second_panda_*")
