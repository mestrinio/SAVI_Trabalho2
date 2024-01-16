import re
import os


pattern = '([a-z_]+)(?=_\d)'

def  switch(filenames):
    name={}
    for filename in filenames:
        match = re.search(pattern, filename)
        label = match.group(1)
        print(label)
        name[label]=[filename]


        # print(label,name[label])
        labels = []
        # print(labels)
        # l=0
        for a in name[label] :
            match a:
                case "apple":
                    labels.append(0)
                case "ball":
                    labels.append(1)
                case "banana":
                    labels.append(2)
                case "bell_pepper":
                    labels.append(3)
                case "binder":
                    labels.append(4)
                case "bowl":
                    labels.append(5)
                case "calculator":
                    labels.append(6)
                case  "camera":
                    labels.append(7)
                case "cap":
                    labels.append(8)
                case "cell_phone":
                    labels.append(9)
                case "cereal_box":
                    labels.append(10)
                case "coffe_mug":
                    labels.append(11)
                case  "comb":
                    labels.append(12)
                case "dry_battery":
                    labels.append(13)
                case "flashlight":
                    labels.append(14)
                case "food_bag":
                    labels.append(15)
                case "food_box":
                    labels.append(16)
                case "food_can":
                    labels.append(17)
                case "food_cup":
                    labels.append(18)
                case "food_jar":
                    labels.append(19)
                case  "garlic":
                    labels.append(20)
                case "glue_stick":
                    labels.append(21)
                case "greens":
                    labels.append(22)
                case "hand_towel":
                    labels.append(23)
                case  "instant_noodles":
                    labels.append(24)
                case "keyboard":
                    labels.append(25)
                case "kleenex":
                    labels.append(26)
                case "lemon":
                    labels.append(27)
                case  "lightbulb":
                    labels.append(28)
                case "lime":
                    labels.append(29)
                case "marker":
                    labels.append(30)
                case "mushroom":
                    labels.append(31)
                case  "notebook":
                    labels.append(32)
                case "onion":
                    labels.append(33)
                case "orange":
                    labels.append(34)
                case "peach":
                    labels.append(35)
                case  "pear":
                    labels.append(36)
                case "pitcher":
                    labels.append(37)
                case "plate":
                    labels.append(38)
                case "pliers":
                    labels.append(39)
                case  "potato":
                    labels.append(40)
                case "rubber_eraser":
                    labels.append(41)
                case "scissors":
                    labels.append(42)
                case "shampoo":
                    labels.append(43)
                case  "soda_can":
                    labels.append(44)
                case "sponge":
                    labels.append(45)
                case "stapler":
                    labels.append(46)
                case "tomato":
                    labels.append(47)
                case  "toothbrush":
                    labels.append(48)
                case "toothpaste":
                    labels.append(49)
                case "watter_bottle":
                    labels.append(50)
                case "":
                    labels.append(51)
        print(labels)
    return labels 