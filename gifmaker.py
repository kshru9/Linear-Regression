import os
import imageio
cur_dir = os.getcwd()
images = []

filenames = []
num_of_iterations = 500
for i in range(0,int(num_of_iterations/5),5):
    filenames.append(cur_dir+'/figures/scplots/'+str(i)+'.png')

for filename in filenames:
    images.append(imageio.imread(filename))

imageio.mimsave(cur_dir+'/gifs/'+'sc.gif', images, duration=0.1)

# images = []
# filenames = []
# num_of_iterations = 1000
# for j in range (0,num_of_iterations,10):
#     filenames.append(cur_dir+'/figures/line_fit/'+str(j+1)+'.png')

# filenames.append(cur_dir+'/figures/line_fit/'+str(num_of_iterations+1)+'.png')

# for filename in filenames:
#     images.append(imageio.imread(filename))

# imageio.mimsave(cur_dir+'/gifs/'+'line_fit.gif', images, duration=0.1)