import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F

class BandFilterLoss():
    def __init__(self, r1, r2):
        self.r1 = r1
        self.r2 = r2

    def __call__(self, target, generated):
        return self.band_filter_loss(target, generated, self.r1, self.r2)
    
    def band_pass_channel(self, channel, mask):
        #transform to frequency domain with FFT
        f = np.fft.fft2(channel)
        f_shift = np.fft.fftshift(f)
        
        #mask specific frequencies
        f_shift_filtered = f_shift * mask
        
        #invert back to get new image
        f_inverse_shift = np.fft.ifftshift(f_shift_filtered)
        channel_filtered = np.fft.ifft2(f_inverse_shift)
        channel_filtered = np.abs(channel_filtered).astype(int)#remove imaginary part, and make sure integers
        return channel_filtered

    def band_pass_filter(self, img,r1,r2):

        # Split the image into color channels
        r = img[:,:,0]
        g = img[:,:,1]
        b = img[:,:,2]
        
        rows, cols, depth = img.shape
        center_row, center_col = rows // 2, cols // 2#get center
        
        # # Create a circular mask
        mask = np.zeros((rows, cols), np.uint8)#get rid of everything
        mask = cv2.circle(mask, (center_col, center_row), r2, 1, -1)#actually keep disk of r2
        if r1>0:
            mask = cv2.circle(mask, (center_col, center_row), r1, 0, -1)    #actually get rid of inner disk again r1 (creating band)
        
        #mask each color
        r_filtered = self.band_pass_channel(r, mask)
        g_filtered = self.band_pass_channel(g, mask)
        b_filtered = self.band_pass_channel(b, mask)

        #combine to one image
        filtered_image = cv2.merge((r_filtered, g_filtered, b_filtered))
        return filtered_image
    
    def band_filter_loss(self, target, generated, r1, r2):
        target_filtered = self.band_pass_filter(target, r1, r2)
        generated_filtered = self.band_pass_filter(generated, r1, r2)
        loss = np.mean(((target_filtered - generated_filtered)/256)**2)
        return loss
    
class FourierHeatMap():
    # def __init__(self,size = 512,sigma=1.0):
    #     x = np.linspace(-(size//2), size//2, size)
    #     y = np.linspace(-(size//2), size//2, size)
    #     xv, yv = np.meshgrid(x, y)

    #     # Create the Gaussian matrix
    #     self.gaussian_matrix = multivariate_normal.pdf(np.dstack((xv, yv)), mean=[0, 0], cov=sigma)
        
    def __call__(self, target, generated):
        return self.fourier_loss(target, generated)
    
    def fourier_spectra(self,img):
        spectra = np.zeros_like(img,dtype=float)
        for i in range(3):
            channel = img[i,:,:]
            # map = np.fft.fftshift(np.fft.fft2(channel))
            map = np.fft.fft2(channel)
            spectra[i,:,:] = np.log(np.abs(map))
        return spectra
    
    def fourier_loss(self, target,generated):
        target_spectra = self.fourier_spectra(target)
        generated_spectra = self.fourier_spectra(generated)
        difference_spectra = target_spectra-generated_spectra
        # difference_spectra/=np.max(difference_spectra)
        return np.mean(difference_spectra**2)
    
class FourierLossTorch(torch.nn.Module):
    def __init__(self):
        super(FourierLossTorch, self).__init__()
        
    def forward(self,target, generated):
        input_fft = torch.fft.fft2(generated)
        target_fft = torch.fft.fft2(target)
        # Compute the loss in the Fourier domain separately for real and imaginary parts
        loss_real = F.mse_loss(input_fft.real, target_fft.real)
        loss_imag = F.mse_loss(input_fft.imag, target_fft.imag)

        # Combine the losses (avg)
        loss = (loss_real + loss_imag) / 2

        # diff = torch.fft.fft2(target-generated)
        # diff = torch.log(torch.abs(diff))
        # diff = diff.real**2 + diff.imag**2
        # diff /=torch.max(diff)#scale down to 0,1 to be consitent with other loss
        # print("Min,Max of Diff:",torch.min(diff).item(),torch.max(diff).item())
        # diff = diff #scale down to 0,1 (ish)
        # loss = torch.log(F.mse_loss(diff, torch.zeros_like(diff)))/16
        # loss = torch.log(torch.mean(diff))
        return loss


class BandFilterLossTorch(torch.nn.Module):
    def __init__(self, r1, r2):
        super(BandFilterLossTorch, self).__init__()
        self.r1 = r1
        self.r2 = r2

    def forward(self, target, generated):
        return self.band_filter_loss(target, generated, self.r1, self.r2)
    
    def band_pass_channel(self, channel, mask):
        #transform to frequency domain with FFT
        f = torch.fft.fftn(channel)
        f_shift = torch.fft.fftshift(f)
        
        #mask specific frequencies
        f_shift_filtered = f_shift * mask
        
        #invert back to get new image
        f_inverse_shift = torch.fft.ifftshift(f_shift_filtered)
        channel_filtered = torch.fft.ifftn(f_inverse_shift)
        channel_filtered = torch.abs(channel_filtered).int()#remove imaginary part, and make sure integers
        return channel_filtered

    def band_pass_filter(self, img,r1,r2):
        # Split the image into color channels
        r = img[0, :, :]
        g = img[1, :, :]
        b = img[2, :, :]
        
        depth, rows, cols= img.shape
        center_row, center_col = rows // 2, cols // 2#get center
        
        # Create a circular mask
        mask = torch.zeros((rows, cols), device=img.device, dtype=torch.float32)
        for y in range(rows):
            for x in range(cols):
                if r1**2 <= (x - center_col)**2 + (y - center_row)**2 <= r2**2:
                    mask[y,x] = 1.0
        
        #mask each color
        r_filtered = self.band_pass_channel(r, mask)
        g_filtered = self.band_pass_channel(g, mask)
        b_filtered = self.band_pass_channel(b, mask)

        #combine to one image
        filtered_image = torch.stack((r_filtered, g_filtered, b_filtered), dim=2)
        return filtered_image
    
    def band_filter_loss(self, target, generated, r1, r2):
        target_filtered = self.band_pass_filter(target, r1, r2)
        generated_filtered = self.band_pass_filter(generated, r1, r2)
        loss = torch.mean(((target_filtered - generated_filtered)/256)**2)
        return loss
        