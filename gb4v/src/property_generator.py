import os
import sys
import random
import argparse
import numpy as np
from functools import partial
from PIL import Image

PROP_EXT = {'planet':'rlv',
            'reluplex':'nnet',
            'mipverify':'jl'}


class PropertyGenerator():

    def __init__(self):
        pass

    def generate(self, verifier, image_path, epsilon, prop_type, label, prop_out_path):
        self.verifier = verifier
        self.image_path = image_path
        self.epsilon = epsilon
        self.prop_type = prop_type
        self.label = label
        self.prop_out_path = prop_out_path

        getattr(self, 'to_'+verifier)()


    def to_planet(self, image_path, target, maxDifferencePerPixel, maxUnsmoothnessInNoise, prop_path, image_size):
        X = image_size
        Y = image_size
        # Read image file
        im = Image.open(image_path) #Can be many different formats.
        im = im.resize((X,Y), Image.ANTIALIAS)
        assert im.size==(X,Y)
        pix = im.load()

        pixels = [pix[x,y] for y in range(0,Y) for x in range(0,X)]

        outtaFile = open(prop_path, "w")

        # Set the boundaries
        for c in range(0, 3):
            for i in range(0,X*Y):
                # Outermost pixels need to be excluded
                x = i % Y
                y = int(i / X)
                if x<3 or x>X-4 or y<3 or y>Y-4:
                    border = 0.0
                else:
                    border = maxDifferencePerPixel

                mini = max(0.0,pixels[i][c]/256.0-border)
                maxi = min(1.0,pixels[i][c]/256.0+border)

                outtaFile.write("Assert <= "+str(mini)+" "+"1.0 inX"+str(i+c*X*Y)+"\n")
                outtaFile.write("Assert >= "+str(maxi)+" "+"1.0 inX"+str(i+c*X*Y)+"\n")
                #print("Assert <= "+str(mini)+" "+"1.0 inX"+str(i+c*X*Y)+"")
                #print("Assert >= "+str(maxi)+" "+"1.0 inX"+str(i+c*X*Y)+"")

        # Set the output
        # TODO: modify this to work for general purpose
        '''
        for i in range(0, 1000):
            if i!=targetDigit:
                outtaFile.write("Assert >= -0.000001 1.0 outX"+str(i)+" -1.0 outX"+str(targetDigit)+"\n")
                #print("Assert >= -0.000001 1.0 outX"+str(i)+" -1.0 outX"+str(targetDigit)+"")
        '''
        # flipped min&max
        #outtaFile.write("Assert >= "+str(target[0])+" 1.0 outX0\n")
        outtaFile.write("Assert <= "+str(target[1])+" 1.0 outX0\n")
        
        if maxUnsmoothnessInNoise<1.0:
            for x in range(0,X):
                for y in range(0,Y):
                    # Smooth down
                    if (y<Y-1):
                        pixelDiff = (pixels[y*Y+x]-pixels[(y+1)*Y+x])/256.0
                        outtaFile.write("Assert <= "+str(pixelDiff-maxUnsmoothnessInNoise)+" 1.0 inX"+str(y*Y+x)+" -1.0 inX"+str((y+1)*X+x)+"\n")
                        outtaFile.write("Assert >= "+str(pixelDiff+maxUnsmoothnessInNoise)+" 1.0 inX"+str(y*Y+x)+" -1.0 inX"+str((y+1)*X+x)+"\n")
                    # Smooth right
                    if (x<X-1):
                        pixelDiff = (pixels[y*Y+x]-pixels[y*Y+x+1])/256.0
                        outtaFile.write("Assert <= "+str(pixelDiff-maxUnsmoothnessInNoise)+" 1.0 inX"+str(y*X+x)+" -1.0 inX"+str(y*X+x+1)+"\n")
                        outtaFile.write("Assert >= "+str(pixelDiff+maxUnsmoothnessInNoise)+" 1.0 inX"+str(y*X+x)+" -1.0 inX"+str(y*X+x+1)+"\n")
        outtaFile.close()

    def old_planet():
        maxDifferencePerPixel = 0.05
        maxUnsmoothnessInNoise = 1
        
        print(true_label)
        
        #min = np.tan((true_label-30*np.pi/180)/2)
        #max = np.tan((true_label+30*np.pi/180)/2)
        #target = (min, max)

        prop_name = "prop_{}-{}-{}-{}-{}-{}.rlv.txt".format(image_name, true_label, target[0], maxDifferencePerPixel, maxUnsmoothnessInNoise, size)
        prop_path = os.path.join(args.output_dir, prop_name)
        print("Generating prop "+prop_name+"...")
       
        self.to_planet(image_path, target, maxDifferencePerPixel, maxUnsmoothnessInNoise, prop_path, size)



        def regression():
        #elif dataset == 'imagenet':

            img_dir = 'img-imagenet-10'
            imgs = [ x for x in os.listdir(img_dir) if "JPEG" in x]
            img_label_file = open('./synset.txt').readlines()
            img_label_dict = {x.strip().split(' ')[0]:i for i,x in enumerate(img_label_file)}
            maxDifferencePerPixel = 0.05
            maxUnsmoothnessInNoise = 1
            img_size = 14

            for i in range(len(imgs)):
                img = imgs[i]
                img_path = os.path.join(img_dir, img)
                true_label = img_label_dict[img.split('_')[0]]

                while True:
                    target_label = np.random.randint(1000)
                    if target_label != true_label:
                        break
                        prop_name = "prop_{}-{}-{}-{}-{}-{}.rlv.txt".format(img, true_label, target, maxDifferencePerPixel, maxUnsmoothnessInNoise, size)

                if verifier == "planet":
                    prop_path = os.path.join(prop_dir, prop_name)
                    gen = partial(gen_prop_rlv, img_path, target_label, maxDifferencePerPixel, maxUnsmoothnessInNoise, prop_path, img_size)
                elif verifier == "reluplex":
                    assert False, "Run the rlv2nnet converter from PLNN on the combination of the network and property rlv file to generate the nnet input format for ReluPlex."
                else:
                    assert False, "Unkown verifier"

                gen()
                print("Generating prop {} of {}.".format(i, len(imgs)))
                print("Evolution complete.")

        #else:
        #    assert False, "Unkown dataset"


    def to_bab():
        self.to_planet()


    def to_babsb(self):
        self.to_planet()


    def to_reluplex(self):
        pass


    def to_mipverify(self):
        
        if self.prop_type == "regression":
            lb,ub = self.label

            lines = []
            lines += ["lp_w = reshape(Array{Float32}([1,0]),(1,2))"]
            lines += ["lp_b = Array{Float32}(["+str(lb)+",0])"]
            lines += ["lp = Linear(lf_w, lf_b)"]
            lines += ['img = load("{}")'.format(self.image_path)]
            lines += ['img = Float64.(cat(3,red(img),green(img),blue(img)))']
            lines += ['img = reshape(img,(1,img_size,img_size,3))']
            lines += ['MIPVerify.find_adversarial_example(nn, img, 1, GurobiSolver(OutputFlag=0), pp = MIPVerify.LInfNormBoundedPerturbationFamily({}), norm_order=Inf);'.format(self.epsilon)]
            lines = [ x +'\n' for x in lines]
            open(self.prop_out_path+'.lb.'+PROP_EXT[self.verifier], 'w').writelines(lines)

            lines = []
            lines += ["lp_w = reshape(Array{Float32}([1,0]),(1,2))"]
            lines += ["lp_b = Array{Float32}(["+str()+",0])"]
            lines += ["lp = Linear(lf_w, lf_b)"]
            lines += ['img = load("{}")'.format(self.image_path)]
            lines += ['img = Float64.(cat(3,red(img),green(img),blue(img)))']
            lines += ['img = reshape(img,(1,img_size,img_size,3))']
            lines += ['MIPVerify.find_adversarial_example(nn, img, 2, GurobiSolver(OutputFlag=0), pp = MIPVerify.LInfNormBoundedPerturbationFamily({}), norm_order=Inf);'.format(self.epsilon)]
            lines = [ x +'\n' for x in lines]
            open(self.prop_out_path+'.ub.'+PROP_EXT[self.verifier], 'w').writelines(lines)


        elif self.prop_type == "classification":
            assert False
        else:
            assert False





    def to_eran(self):
        pass


    def to_cnncert(self):
        pass

