import numpy as np
import imageio.v3 as iio

def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3
    
    # Calculate required oversized shape that is exactly divisible by res
    h = ((shape[0] + res[0] - 1) // res[0]) * res[0]
    w = ((shape[1] + res[1] - 1) // res[1]) * res[1]
    
    delta = (res[0] / h, res[1] / w)
    d = (h // res[0], w // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:  ,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:  ].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:  ,1:  ].repeat(d[0], 0).repeat(d[1], 1)
    
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)

    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    noise = np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)
    
    # Crop to requested shape
    return noise[:shape[0], :shape[1]]

def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    # Normalize to 0-1
    noise = (noise - np.min(noise)) / (np.ptp(noise) + 1e-5)
    return noise

def add_synthetic_fog(img, beta=1.5, atm_light=255, noise_scale=2, pattern='patchy'):
    """
    Adds synthetic fog to an image.
    Assumes img is RGB uint8.
    pattern: 'uniform', 'patchy', or 'gradient'
    """
    h, w = img.shape[:2]
    img = img.astype(np.float32)
    
    if pattern == 'uniform':
        depth = np.ones((h, w), dtype=np.float32) * 0.8
    elif pattern == 'gradient':
        depth = np.linspace(0.1, 1.0, h)[:, None]
        depth = np.tile(depth, (1, w)).astype(np.float32)
    elif pattern == 'patchy':
        res = (max(1, h//100), max(1, w//100))
        noise = generate_fractal_noise_2d((h, w), res, octaves=3)
        depth = 0.3 + 0.7 * noise
    
    # Transmission map
    trans = np.exp(-beta * depth)
    trans = np.stack([trans]*3, axis=-1)
    
    # Apply atmospheric scattering model
    hazy = img * trans + atm_light * (1 - trans)
    hazy = np.clip(hazy, 0, 255).astype(np.uint8)
    
    return hazy, trans[:,:,0]

if __name__ == "__main__":
    # Test generation — output is RGB
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[:] = (50, 150, 50)  # Green background
    
    hazy, trans = add_synthetic_fog(img, pattern='patchy')
    iio.imwrite("synthetic_haze_test.jpg", hazy)
    print("Test image generated.")
