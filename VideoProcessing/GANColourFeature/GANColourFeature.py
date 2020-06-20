def cooccurence_2D_matrix(img, symm):
    
    import cv2
    import numpy as np
    
    
    I_x = img[1:-1,1:-1]
    
    I_n = [img[1:-1,2:],
          img[1:-1,:-2],
          img[2:,1:-1],
          img[:-2,1:-1]]
    
    clips = np.array([[-np.inf,-2], [-1, -1], [0, 0],[1, 1], [2, np.inf]])
    N = clips.shape[0]
    P = np.zeros([N,N,N])
    
    for j in range(0, len(I_n)):
        D = I_x - I_n[j]
        D_ = np.zeros(D.shape)
        
        for k in range(0, np.size(D_)):
            
            row = np.unravel_index([k], D_.shape)[0][0]
            column = np.unravel_index([k], D_.shape)[1][0]
        
            for i in range(0,N):
                D_[D[row,column]>=clips[i,0] and D[row,column]<=clips[i,1]] = i
  
        D_1 = D_[:,:-2]
        D_2 = D_[:,1:-1]
        D_3 = D_[:,2:]
        
        M = np.zeros([N,N,N])
    
        for k in range(0, np.size(D_1)):
            
            row = np.unravel_index([k], D_1.shape)[0][0]
            column = np.unravel_index([k], D_1.shape)[1][0]
            
            
            if symm and D_1[row,column] > D_3[row,column]:
                M[D_3[row,column].astype(int),D_2[row,column].astype(int),D_1[row,column].astype(int)] = M[D_3[row,column].astype(int),D_2[row,column].astype(int),D_1[row,column].astype(int)] + 1
                
            else:
                M[D_1[row,column].astype(int),D_2[row,column].astype(int),D_3[row,column].astype(int)] = M[D_1[row,column].astype(int),D_2[row,column].astype(int),D_3[row,column].astype(int)] +1



        P = P + M/np.size(D_1)
        
        
        D_1 = D_[:-2,:]
        D_2 = D_[1:-1,:]
        D_3 = D_[2:,:]
        M = np.zeros([N,N,N])
        
        for k in range(0,np.size(D_1)):
            
            row = np.unravel_index([k], D_1.shape)[0][0]
            column = np.unravel_index([k], D_1.shape)[1][0]
            
            if symm and D_1[row,column] > D_3[row,column]:
                M[D_3[row,column].astype(int),D_2[row,column].astype(int),D_1[row,column].astype(int)] = M[D_3[row,column].astype(int),D_2[row,column].astype(int),D_1[row,column].astype(int)] + 1
                
            else:
                M[D_1[row,column].astype(int),D_2[row,column].astype(int),D_3[row,column].astype(int)] = M[D_1[row,column].astype(int),D_2[row,column].astype(int),D_3[row,column].astype(int)] +1


        P = P + M/np.size(D_1)
        
    if symm:
        P_ = np.zeros([N,N,N])
            
        for i in range(0,N):
            for j in range(0,N):
                for k in range(0,N):
                    if i <= k :
                        P_[i,j,k] = 1
                            
        P_ = P_.astype(bool)                    
        F = P[P_].conj().T
            
    else:
        F = P[:].conj().T
        
    return F      


#########################################
#########################################

def cooccurence_3D_matrix(img, symm):
    
    import cv2
    import numpy as np
    
    
    I_x = img[1:-1,1:-1]
    
    I_n = [img[1:-1,2:],
          img[1:-1,:-2],
          img[2:,1:-1],
          img[:-2,1:-1]]
    
    base = 2 
    N = base**3
    P = np.zeros([N,N,N])
    
    for j in range(0, len(I_n)):
        D = I_x > I_n[j]
        D_ = D[:,:,0]*base**0 + D[:,:,1]*base**1 + D[:,:,2]*base**2 
        
            
        D_1 = D_[:,:-2]
        D_2 = D_[:,1:-1]
        D_3 = D_[:,2:]
        
        M = np.zeros([N,N,N])
        
        for k in range(0, np.size(D_1)):
                    
            row = np.unravel_index([k], D_1.shape)[0][0]
            column = np.unravel_index([k], D_1.shape)[1][0]
            
            if symm and D_1[row,column] > D_3[row,column]:
                M[D_3[row,column],D_2[row,column],D_1[row,column]] = M[D_3[row,column],D_2[row,column],D_1[row,column]] + 1
                
            else:
                M[D_1[row,column],D_2[row,column],D_3[row,column]] = M[D_1[row,column],D_2[row,column],D_3[row,column]] +1
    
        P = P + M/np.size(D_1)
        
        
        D_1 = D_[:-2,:]
        D_2 = D_[1:-1,:]
        D_3 = D_[2:,:]
        M = np.zeros([N,N,N])
        
        for k in range(0,np.size(D_1)):
            
            row = np.unravel_index([k], D_1.shape)[0][0]
            column = np.unravel_index([k], D_1.shape)[1][0]
            
            if symm and D_1[row,column] > D_3[row,column]:
                M[D_3[row,column],D_2[row,column],D_1[row,column]] = M[D_3[row,column],D_2[row,column],D_1[row,column]] + 1
            else:
                M[D_1[row,column],D_2[row,column],D_3[row,column]] = M[D_1[row,column],D_2[row,column],D_3[row,column]] + 1
                
        
        P = P + M/np.size(D_1)
        
        
    if symm:
        P_ = np.zeros([N,N,N])
            
        for i in range(0,N):
            for j in range(0,N):
                for k in range(0,N):
                    if i <= k :
                        P_[i,j,k] = 1
                            
        
        P_ = P_.astype(bool)
        F = P[P_].conj().T
            
    else:
        F = P[:].conj().T
                
    return F 

######################
######################

def GAN_COLOUR_FEATURE(image):
    """
    Extract features for detecting GAN generated images.
    
    Input: 
    - image: path to the image file.
    Output:
    - The 588-D feature for GAN generated image detection.
    
    """
    import cv2
    import numpy as np
    
    
    img = cv2.imread(image)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    ycc = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    
    Feature = np.concatenate([cooccurence_3D_matrix(img, True),
                             cooccurence_2D_matrix(hsv[:,:,0],True),
                             cooccurence_2D_matrix(hsv[:,:,1],True),
                             cooccurence_2D_matrix(ycc[:,:,1],True),
                             cooccurence_2D_matrix(ycc[:,:,2],True)],
                             axis = None)
    
    return Feature