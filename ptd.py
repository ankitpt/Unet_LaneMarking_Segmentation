 cnt=0;    
    for i in range(rows):
        
        for j in range(cols):
            
          
            try:
                data2=data[loc[i,j]]
                val=np.mean(data2[:,5])
                img[i,j]=val
                cnt=cnt+1
            except:
              
              continue
            
    nfilename=str(num)+".png"

    os.chdir("//myhome.itap.purdue.edu/myhome/pate1019/ecn.data/Desktop/DPRG_ECN/Pt_density")
    
    img2 = Image.fromarray(img.astype(np.uint8))
    img2 = img2.resize((256,256))
    img2=img2.rotate(180)
    img2.transpose(Image.FLIP_LEFT_RIGHT).save(nfilename,"PNG")
   # plt.imsave(nfilename, img2, cmap='gray')
    print("Image saved,",nfilename)
    os.chdir("//myhome.itap.purdue.edu/myhome/pate1019/ecn.data/Desktop/DPRG_ECN/Point_Clouds6_train")