import logging
import glob
import re
import os

import numpy as np
import matplotlib.pyplot as plt

import scipy.io

from scipy.ndimage import median_filter
from scipy.signal import butter, lfilter

class SondaAnalyze():
    def __init__(self, path):
        """
        Initialize the SondaAnalyze object with the given path.
        """
        self.__data = []
        self.__dataMatrix = []
        self.__displacementX = []
        self.__displacementZ = []
        self.__freq = []
        self.__InitData = None
        self.__meanDisplacement = []
        self.__maxDisplacement = []
        self.testedROI = False
        self.wROI = False
        self.__importMat(path)
        if not self.__displacementX:
            self.getDx()
        if not self.__displacementZ:
            self.getDz()
        if not self.__dataMatrix:
            self.getDispMap()
        if not self.__freq:
            self.getFreq()

    def __importMat(self, path):
        """
        Import .mat files from a given path and add the data to the matrix.
    
        Args:
            path (str): The path to the folder containing the .mat files.
        
        Raises:
            FileNotFoundError: If the path does not exist.
            Exception: If there is an error while reading a file.
        """
        # Verifica se o caminho da pasta existe
        if not os.path.exists(path):
            raise FileNotFoundError(f'O caminho da pasta {path} não existe.')
    
        # Itera sobre os arquivos .mat na pasta
        for file_path in glob.glob(os.path.join(path, '*.mat')):
            # Carrega o arquivo .mat e adiciona os dados à matriz
            try:
                data = scipy.io.loadmat(file_path)
                filename = os.path.basename(file_path)
                match = re.search(r'\d+Hz', filename)

                if match:
                    freq = match.group()
                    data['Freq'] = freq  # Adicionando a nova frequência à lista existente
                self.__data.append(data)
                logging.info(f'Arquivo {filename} lido com sucesso.')
            except scipy.io.matlab.miobase.MatReadError as e:
                raise Exception(f'Erro ao ler o arquivo {filename}: {e}')

    def getDispMap(self):
        """
        Returns the disp_map data matrix.
        """
        if not self.__dataMatrix:
            for data in self.__data:
                self.__dataMatrix.append(np.array(data.get('disp_map')))
                self.__dataMatrix[-1] = np.transpose(self.__dataMatrix[-1], axes=(2, 0, 1))
        return list(self.__dataMatrix)

    def getDx(self):
        """
        Get the displacement in the x-direction.
        """
        if self.__data and not self.__displacementX:
            # Suggestion 1: Use a more descriptive variable name instead of dx
            self.__displacementX = [data.get('dx') for data in self.__data]
        # Suggestion 2: Check if self.__data is not empty before checking if self.__dx is not empty
        return self.__displacementX if self.__displacementX else self.__displacementX

    def getDz(self):
        """
        Returns the list of 'dz' values from the data dictionary.
        """
        if not self.__displacementZ:
            self.__displacementZ = [data.get('dz', None) for data in self.__data]
        return self.__displacementZ

    def getFreq(self):
        """
        Returns the list of frequencies from the data dictionary.

        Returns:
            list: The list of frequencies.
        """
        if not self.__freq and self.__data:
            self.__freq = [data.get('Freq') for data in self.__data]
        return self.__freq

    def getMax(self):
        """
        Returns the maximum value of the displacement.
        """
        if self.__max is None:
            return None
        return self.__max

    def getMean(self):
        """
        Returns the mean displacement.

        Returns:
            float: The mean displacement.
        """
        if self.__med is None:
            return None
        return self.__med

    def setLowPassFilter(self, cutOff, fs, order = 5):
        if fs <= 0:
            raise ValueError("fs must be positive.")
        if cutOff > 0.5 * fs:
            raise ValueError("The cutoff frequency is greater than the Nyquist frequency, which can cause aliasing.")

        nyquist = 0.5 * fs
        normal_cutoff = cutOff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

        with np.nditer(self.__dataMatrix, flags=['multi_index'], op_flags=['readwrite']) as it:
            for data in it:
                y = lfilter(b, a, data, axis=0)
                data[...] = np.array(y)

    def setMedianFilter(self, cond: bool, size=5):
        """
        Apply median filter to the data matrix.

        Args:
            cond (bool): Condition to apply the median filter.
            size (int): Size of the median filter.

        Returns:
            None
        """
        if self.__InitData is None:
            raise Exception("No initial data available.")
        else:
            self.__dataMatrix = self.__InitData

        if cond:
            for i in range(len(self.__dataMatrix)):
                for j in range(len(self.__dataMatrix[i])):
                    self.__dataMatrix[i][j] = median_filter(np.array(self.__dataMatrix[i][j]), size = size)

    def SelectPlot(self, index, region):
        """
        Selects and plots data from self.__dataMatrix based on the given index and region.

        Args:
            index (int): The index of the data to be plotted.
            region (list): The list of regions to be plotted.

        Returns:
            None
        """
        data_matrix = self.__dataMatrix[index]
        if data_matrix is not None:
            fig, axs = plt.subplots(len(region), 2, figsize=(12, 6 * len(region)))
            for (i, r) in enumerate(region):
                im1 = axs[i, 0].imshow(abs(data_matrix[r]), cmap='viridis', aspect='auto')
                axs[i, 0].set_title('Absolute Displacement Plot')
                fig.colorbar(im1, ax=axs[i, 0])

                im2 = axs[i, 1].imshow(data_matrix[r], cmap='plasma', aspect='auto')
                axs[i, 1].set_title('Raw Displacement Plot')
                fig.colorbar(im2, ax=axs[i, 1])
            plt.subplots_adjust(wspace=0.5)
            plt.show()

    def TestSetRectROI(self, number, ind_time, x, x0, z, z0):
        """
        This method sets a rectangular region of interest (ROI) for testing purposes.
    
        Args:
            number (int): The index of the data matrix to be used.
            ind_time (int): The index of the time dimension to be used.
            x (int): The maximum x-coordinate of the ROI.
            x0 (int): The minimum x-coordinate of the ROI.
            z (int): The maximum y-coordinate of the ROI.
            z0 (int): The minimum y-coordinate of the ROI.
    
        Raises:
            ValueError: If the number is invalid or out of range.
            ValueError: If the indices are out of range.
        """
        if number < 0 or number >= len(self.__dataMatrix):
            raise ValueError("Invalid number. It should be within the range of self.__dataMatrix.")
    
        plt.close('all')
    
        if self.__dataMatrix is not None:
            img = self.__dataMatrix[number]
        
            if img is not None and ind_time < img.shape[0] and x0 < x and z0 < z:
                img2 = img[ind_time][z0:z, x0:x]
                plt.imshow(img2, cmap='viridis')
                plt.colorbar()
                plt.show(block=False)

                self.testedROI = True

            else:
                raise ValueError("Indices are out of range.")

    def setRectROI(self, max_x, min_x, max_z, min_z):
        """
        Sets the rectangular region of interest (ROI) based on the given coordinates.

        Args:
            max_x (int): The maximum x-coordinate of the ROI.
            min_x (int): The minimum x-coordinate of the ROI.
            max_z (int): The maximum y-coordinate of the ROI.
            min_z (int): The minimum y-coordinate of the ROI.
        """
        self.__dataMatrixROI = []
        if self.testedROI and self.__dataMatrix is not None:
            for i in range(len(self.__dataMatrix)):
                inter = []
                for j in range(self.__dataMatrix[i].shape[0]):
                    inter.append(self.__dataMatrix[i][j][min_z:max_z, min_x:max_x])
                self.__dataMatrixROI.append(np.array(inter))
        else:
            raise Exception("Realizar teste de ROI")
        self.wROI = True

    def setRectBG(self, x, x0, z, z0):
        if x < x0 or z < z0:
            raise ValueError("Invalid indices. The maximum indices should be greater than the minimum indices.")
        self.__BG = []
        if self.__dataMatrix is None or len(self.__dataMatrix) == 0:
            return
        for data in self.__dataMatrix:
            if x >= data.shape[2] or z >= data.shape[1]:
                raise ValueError("Invalid indices. The indices are out of range for the data matrix.")
            inter = []
            data_matrix = data
            for data_ in data_matrix:
                inter.append(np.mean(abs(data_[z0:z, x0:x])))
            self.__BG.append(np.array(inter))

    def MeanTempPlot(self):
        if self.wROI:
            data = self.__dataMatrixROI
        else:
            raise Exception("Selecione a ROI")

        abs_data = np.abs(data)
        med_array = np.mean(abs_data, axis=(2,3))
        std_array = np.std(abs_data, axis=(2,3))

        index_range = range(len(med_array))
        freq = self.__freq

        for i in index_range:
            plt.scatter(range(med_array[i].shape[0]), med_array[i])
            plt.errorbar(range(med_array[i].shape[0]), med_array[i], yerr=std_array[i], linestyle='None', capsize=5,
                               label=freq[i])

        plt.xlabel('Tempo')
        plt.ylabel('Deslocamento Medio(um)')
        plt.legend()
        plt.title("w/ ROI")
        plt.show()

        max = []
        mean = []
        error = []
        for i in index_range:
            max.append(np.max(med_array[i]))
            mean.append(np.mean(med_array[i]))
            error.append(np.std(med_array[i]))

        plt.scatter(freq, max)
        plt.xlabel('Frequencia')
        plt.ylabel('Deslocamento')
        plt.legend()
        plt.title("Deslocamento Maximo")

        plt.show()

        plt.scatter(freq, mean)
        plt.errorbar(freq, mean, yerr=error, linestyle='None', capsize=5)
        plt.xlabel('Frequencia')
        plt.ylabel('Deslocamento')
        plt.legend()
        plt.title("Deslocamento Medio")

        plt.show()

        return med_array, max, mean

    def ContrastPlotTemp(self):
        if len(self.__BG) != 0 and self.wROI:
            data = self.__dataMatrixROI
            BG = self.__BG
            contrast_array = []
            j = 0
            for data_full in data:
                med = []

                for i, data_ in enumerate(data_full):
                    med.append((np.mean(abs(data_))-BG[j][i])/BG[j][i])
                contrast_array.append(np.array(med))
                j = j + 1

            for i in range(len(contrast_array)):
                plt.scatter(range(len(contrast_array[i])), contrast_array[i], label = self.__freq[i])

            plt.xlabel("Tempo [index]")
            plt.ylabel("Contraste")
            plt.legend()
            plt.show()
        
            max = []
            mean = []
            for contrast in contrast_array:
                max.append(np.max(contrast))
                mean.append(np.mean(contrast))

            plt.scatter(self.__freq, max)
            plt.xlabel('Frequencia')
            plt.ylabel("Contraste")
            plt.title("Maximo Contraste")
            plt.show()
            plt.scatter(self.__freq, mean)
            plt.xlabel('Frequencia')
            plt.ylabel('Contraste')
            plt.title("Contraste Medio")
            plt.show()
        
            return contrast_array, max, mean
        else:
            raise Exception("Selecione o background")

    def sortDataFreq(self, sequence):
        if not isinstance(sequence, (list, tuple, set)):
            sequence = list(sequence)

        if len(self.__dataMatrix) != len(self.__freq):
            raise ValueError("The lengths of `self.__dataMatrix` and `self.__freq` are different.")

        if any(i < 0 or i >= len(self.__dataMatrix) for i in sequence):
            raise ValueError("Invalid indices in `sequence`.")

        sortData = []
        sortFreq = []

        for i in sequence:
            sortData.append(self.__dataMatrix[i])
            sortFreq.append(self.__freq[i])

        self.__dataMatrix = sortData
        self.__freq = sortFreq

        return True