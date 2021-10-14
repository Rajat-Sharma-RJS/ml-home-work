import numpy as np
import cv2
import matplotlib.pyplot as plt

def importing_images(name, num, each, pre):
    folder = name
    images = []
    
    for i in range(num):
        sub = "s"+str(i+1)
        store = []
        for j in range(each):
            file = str(pre+j+1)+".pgm"
            img = cv2.imread(folder+"/"+sub+"/"+file, 0)
            store.append(img)
        store = np.array(store)
        
        flag = False
        ent = None
        for pic in store:
            if flag == False:
                ent = pic.copy()
                flag = True
            else:
                ent = ent+pic
        
        vec = (1/each)*ent
        images.append(vec)
    
    images = np.array(images)
    return images

def converting(images):
    picture = []
    for img in images:
        ent = []
        for row in img:
            for col in row:
                ent.append(col)
        ent = np.array(ent)
        picture.append(ent)
    picture = np.array(picture)
    return picture

def collect_images(name, num, each, pre):
    folder = name
    images = []
    
    for i in range(num):
        sub = "s"+str(i+1)
        for j in range(each):
            file = str(pre+j+1)+".pgm"
            img = cv2.imread(folder+"/"+sub+"/"+file, 0)
            images.append(img)
    
    images = np.array(images)
    return images

def averaging(images):
    picture = images.copy()
    flag = False
    avg = None
    for img in picture:
        if flag == False:
            avg = img.copy()
            flag = True
        else:
            avg = avg+img
    
    avg = (1/len(picture))*avg
    
    for i in range(len(picture)):
        picture[i] = picture[i] - avg
    return picture

def eigenFace(eigValue, eigVector, K, training_data):
    eigHash = dict()
    for i in range(len(eigValue)):
        eigHash.update({eigValue[i]: eigVector[:,i]})
    
    feature = np.zeros((len(eigValue), K))
    vec = sorted(eigHash.items(), key = lambda x: (x[0], x[1]), reverse=True)
    for i in range(K):
        feature[:,i] = vec[i][1]
    
    eigFace = np.matmul(feature.T, training_data)
    return eigFace

def sampleEigen(eigFace, M, N):
    eigImg = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            eigImg[i, j] = eigFace[N*i+j]
    
    return eigImg

def prediction(eigSig, projected):
    output = []
    row = projected.shape[0]
    col = projected.shape[1]
    num = eigSig.shape[1]
    
    for i in range(col):
        img = projected[:,i].copy()
        flag = False
        least = None
        x = None
        for n in range(num):
            pic = eigSig[:,n].copy()
            ## euclid distance
            dist = np.sqrt(((pic - img)**2).sum())
            if flag == False:
                least = dist
                x = n
                flag = True
            else:
                if dist < least:
                    least = dist
                    x = n
        output.append(x)
    
    output = np.array(output)
    return output

def accuracy(output, actual):
    acc = 0
    for i in range(len(actual)):
        if actual[i] == output[i]:
            acc += 1
    per = round((acc/len(actual))*100, 2)
    return per

def studyingEffect(eigValue, eigVector, images, testing, I, J):
    kv, ka = [], []
    for i in range(len(eigValue)):
        k = i+1
        eigFace = eigenFace(eigValue, eigVector, k, images)
        eigSig = np.matmul(eigFace, images.T)
        projected = np.matmul(eigFace, testing.T)
        output = prediction(eigSig, projected)
        actual = []
        for i in range(I):
            for j in range(10-J):
                actual.append(i)
        actual = np.array(actual)
        acc = accuracy(output, actual)
        ka.append(acc)
        kv.append(k)
    kv = np.array(kv)
    ka = np.array(ka)
    return (kv, ka)

I, J, K = 13, 3, 13
print("\nTraining Images\n")
training_images = importing_images("Face_Data", I, J, 0)
processed = averaging(training_images)
training_data = converting(processed)
covariance = np.matmul(training_data, training_data.T)
print("Covariance(for n = 13) :\n", covariance)
eigValue, eigVector = np.linalg.eig(covariance)
print("\nEigen Values(for n = 13) :\n",eigValue)
print("\nEigen Vectors(for n = 13) :\n", eigVector.T)
eigFace = eigenFace(eigValue, eigVector, K, training_data)
M, N = processed[0].shape[0], processed[0].shape[1]
eigImg = sampleEigen(eigFace[0], M, N)
eigSig = np.matmul(eigFace, training_data.T)
print("\nTesting Images\n")
testing_image = collect_images("Face_Data", I, 10-J, J)
process_test = averaging(testing_image)
testing_data = converting(process_test)
projected = np.matmul(eigFace, testing_data.T)

kv, ka = studyingEffect(eigValue, eigVector, training_data, testing_data, I, J)
output = prediction(eigSig, projected)
actual = []
for i in range(I):
    for j in range(10-J):
        actual.append(i)
actual = np.array(actual)
ka = ka*3
print("Generating Graph")

fig, axis = plt.subplots(2, 2, figsize=(8,8))
axis[0][0].imshow(training_images[1])
axis[0][0].set_title("Training Image")
axis[0][1].imshow(processed[1])
axis[0][1].set_title("After Averaging")
axis[1][0].imshow(eigImg)
axis[1][0].set_title("Eigen Face")
axis[1][1].plot(kv, ka, 'g-')
axis[1][1].set_title("Accuracy v/s K")
axis[1][1].set_xlabel("K(for n=13)")
axis[1][1].set_ylabel("Accuracy (%)")

plt.show()
