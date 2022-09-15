import paddlehub as hub
from matplotlib import pyplot as plt
module = hub.Module(name="ernie_vilg")
results = module.generate_image(text_prompts=["一只黑色的小狗在开火车"])
for i in results:
    plt.imshow(i)
    plt.show()
