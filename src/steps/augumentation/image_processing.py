
def compute_gamma(image):
    avg_channel = np.mean(image[:,:,0:2], axis=2)
    return np.median(avg_channel)

gamma_max = 200
def gammaCorrection(image, gamma):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(image, lookUpTable)
    img_gamma_corrected = cv2.hconcat([image, res])
    return img_gamma_corrected

TARGET_GAMMA = 1.8

def auto_gamma_correction(image):
    x = 0.2 / 40
    b = 0.5375
    g = compute_gamma(image)
    compensation_gamma = x * g * TARGET_GAMMA + b
    return gammaCorrection(image, compensation_gamma)
