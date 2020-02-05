
    def gen_prop():
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
        outtaFile.write("Assert <= "+str(targetDigit[1])+" 1.0 outX0\n")
        #outtaFile.write("Assert >= "+str(targetDigit[0])+" 1.0 outX0\n")
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


def main():

    true_label = 0
    target_label = 10
    maxDifferencePerPixel =0.05
    maxUnsmoothnessInNoise = 1
    prop_name = "prop_{}-{}-{}-{}-{}.rlv.txt".format(img, true_label, target_label, maxDifferencePerPixel, maxUnsmoothnessInNoise)

    image_path = os.path.join(img_dir, img)
    prop_path = os.path.join(prop_dir, prop_name)
    gen_prop_rlv(image_path, target_label, maxDifferencePerPixel, maxUnsmoothnessInNoise, prop_path, 28, 28)
