import tkinter as tk
from tkinter import Label, ttk, filedialog, messagebox
from pathlib import Path
import numpy as np
from PIL import ImageTk, Image
import pytesseract as tess
from image_processing_tool.contrast import linear_transformation, saturated_transformation
from image_processing_tool.filters import filer_average, filer_median, noise, signal_to_Noise_Ratio
from image_processing_tool.imageIO import readImagePgm 
import matplotlib.pyplot as plt
from image_processing_tool.imageStats import histogram, histogram_equalization, mean_stdev,  histogram_cummulated
from image_processing_tool.thresholding import closing, dilatation, errosion, opening, otsu, thresholding

class Page(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("Image Processing Tool")
        
        container=tk.Frame(self)
        container.grid()

        # menu
        self.option_add('*tearOff', False)
        menubar = tk.Menu(self)
        self.config(menu = menubar)
        file = tk.Menu(menubar)
        stats = tk.Menu(menubar)
        initialImage = tk.Menu(menubar)
        equalization = tk.Menu(menubar)
        linearTrasform = tk.Menu(menubar)
        filters = tk.Menu(menubar)
        thresholding = tk.Menu(menubar)
        morphological_operations = tk.Menu(menubar)


        menubar.add_cascade(menu = file, label = "File")
        file.add_command(label = 'Open...', command=lambda:self.show_image())

        menubar.add_cascade(menu = exit, label = "exit",  command=lambda:self.destroy)

        menubar.add_cascade(menu = stats, label = "Stats")
        stats.add_command(label = 'Mean', command=lambda:self.showMean())
        stats.add_command(label = ' Standard Deviation', command=lambda:self.showStDev())
        stats.add_command(label = 'Histogram', command=lambda:self.show_histogram())
        stats.add_command(label = 'Cumulated Histogram', command=lambda:self.show_histogram_cummulated())

        menubar.add_command( label = "initial Image",command=lambda:self.show_initial_image())

        menubar.add_cascade(menu = equalization, label = "Equalization")
        equalization.add_command(label = 'Equalised Image', command=lambda:self.show_equalised_image())
        equalization.add_command(label = 'Histogram Equalization', command=lambda:self.show_histogram_equalised())

        menubar.add_cascade(menu = linearTrasform, label = "Trasformations")
        linearTrasform.add_command(label = 'Satured Transformation', command=lambda:self.showSaturedTrans())
        linearTrasform.add_command(label = 'Linear Trasformation', command=lambda:self.showLinearTrans())

        menubar.add_cascade(menu = filters, label = "Filters")
        filters.add_command(label = 'Image Noise', command=lambda:self.showImageNoise())
        filters.add_command(label = 'Average Filter', command=lambda:self.showAverageFilter())
        filters.add_command(label = 'Median Filter', command=lambda:self.showMedianFilter())
        filters.add_command(label = 'SNR Filters Ratio', command=lambda:self.showSNR_ratio())

        menubar.add_cascade(menu = thresholding, label = "Thersholding")
        thresholding.add_command(label = 'Thersholding 25', command=lambda:self.showThresh25())
        thresholding.add_command(label = 'Hisotgram Thersholding 25', command=lambda:self.showHistoThresh25())
        thresholding.add_command(label = 'Thersholding 50', command=lambda:self.showThresh50())
        thresholding.add_command(label = 'Histogram Thersholding 50', command=lambda:self.showHistoThresh50())
        thresholding.add_command(label = 'Thersholding 120', command=lambda:self.showThresh120())
        thresholding.add_command(label = 'Histogram Thersholding 120', command=lambda:self.showHistoThresh120())
        thresholding.add_command(label = 'Mean Intensity Thersholding', command=lambda:self.showThreshMeanIntensity())
        thresholding.add_command(label = 'Mean Intensity Histogram Thersholding', command=lambda:self.showHistoThreshMeanIntensity())
        thresholding.add_command(label = 'Otsu Thersholding', command=lambda:self.showThreshOtsu())
        thresholding.add_command(label = 'Otsu Histogram Thersholding', command=lambda:self.showHistoThreshOtsu())

        menubar.add_cascade(menu = morphological_operations, label = "Morphological Operations")
        morphological_operations.add_command(label = 'Dilatation', command=lambda:self.showDilatation())
        morphological_operations.add_command(label = 'Dilatation Histo', command=lambda:self.showDilatationHisto())
        morphological_operations.add_command(label = 'Errosion', command=lambda:self.showErrosion())
        morphological_operations.add_command(label = 'Errosion Histo', command=lambda:self.showErrosionHisto())
        morphological_operations.add_command(label = 'Opening', command=lambda:self.showOpening())
        morphological_operations.add_command(label = 'Opening Histo', command=lambda:self.showOpeningHisto())
        morphological_operations.add_command(label = 'Closing', command=lambda:self.showClosing())
        morphological_operations.add_command(label = 'Closing Histo', command=lambda:self.showClosingHisto())
  
     
     
     
        # title
        self.empty_name=tk.Label(self, text="Image Processing Tool", font=("Arial", 16))
        self.empty_name.grid(row=0, column=0, pady=5, padx=10, sticky="sw")

        # intro
        self.intro_lbl = tk.Label(self, text="Welcome",font=("Arial", 11), fg="#202020")
        self.intro_lbl.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="nw")

        # select image file                        
        self.browse_lbl = tk.Label(self, text="Select Image :", font=("Arial", 10), fg="#202020")
        self.browse_lbl.grid(row=4, column=0, columnspan=3, padx=24, pady=10, sticky="w")

        self.browse_entry=tk.Entry(self, text="", width=30)
        self.browse_entry.grid(row=4, column=0, columnspan=3, padx=120, pady=10, sticky="w")

        self.browse_btn = tk.Button(self, text="     Browse     ", bg="#ffffff", relief="flat", width=10, command=lambda:self.show_image())
        self.browse_btn.grid(row=4, column=0, padx=310, pady=10, columnspan=3, sticky="w")
        
        # file info
        self.lbl_filename = tk.Label(self, text="File Name: ", font=("Arial", 10), fg="#202020")
        self.lbl_filesize = tk.Label(self, text="File Size: ", font=("Arial", 10), fg="#202020") 

        self.label_text_x = tk.StringVar()
        self.lbl_filename_01 = tk.Label(self, textvariable=self.label_text_x, font=("Arial", 10),fg="#202020")
        
        self.text_file_size=tk.StringVar()
        self.lbl_filesize_01 = tk.Label(self, textvariable=self.text_file_size, font=("Arial", 10), fg="#202020")
        
        # place holder for document thumbnail
        self.lbl_image = tk.Label(self, image="")
        self.lbl_image.grid(row=8, column=0, pady=25, padx=10, columnspan=3, sticky="nw")

        # status text
        self.label_text_progress = tk.StringVar()
        self.scan_progress = tk.Label(self, textvariable=self.label_text_progress, font=("Arial", 10),fg="#0000ff")
        self.test=20
        # scan button
        self.scan_btn = tk.Button(self, text="     Process     ", bg="#ffffff", relief="flat", width=10, command=lambda:self.ocr())
        # clear ocr text button
        self.clear_btn = tk.Button(self, text="     Clear      ", bg="#ffffff", relief="flat", width=10, command=lambda:self.clearOcr())
        # text area to place text
        self.ocr_text = tk.Text(self, height=25, width=38)  
       
    
    def show_image(self):
        global path
        global matrix 
        # open file dialog
        path = self.path = filedialog.askopenfilename(defaultextension="PGM", filetypes = (("PGM","*.pgm"),("JPG", "*.jpg"),("PNG","*.png")))
        self.browse_entry.delete(0, tk.END)
        self.browse_entry.insert(0, self.path)
        
        self.label_text_progress.set("Image loaded - ready to be processed.")
        self.scan_progress.grid(row=18, column=0, padx=10, pady=0,columnspan=3, sticky="w")

        #read image
        matrix = readImagePgm(self.path)
        photo = ImageTk.PhotoImage(image = Image.fromarray(matrix))
        #show image
        self.lbl_image.configure(image=photo)
        self.lbl_image.image=photo
        self.lbl_image.grid(padx=350)   

    def about(self):
        # show about message
        messagebox.showinfo(title = 'About', message = 'This is a demo Tkinter project by Rick Torzynski')
    
    def showMean(self):
        meanValue, stdevValue = mean_stdev(matrix)
        self.label_text_progress.set("Mean Value "+str(meanValue))

    def showStDev(self):
        meanValue, stdevValue = mean_stdev(matrix)
        self.label_text_progress.set("Standard Deviation Value "+str(stdevValue))

    def show_histogram(self):
        histo = histogram(matrix)
        plt.bar(range(256), histo)
        plt.xlabel('Graylevel / intensity')
        plt.ylabel('Frequency') 
        plt.show()
        self.label_text_progress.set("Image Histogram")  

    def show_histogram_cummulated(self):
            histo = histogram(matrix)
            histoC = histogram_cummulated(histo)
            plt.bar(range(256), histoC)
            plt.xlabel('Graylevel / intensity')
            plt.ylabel('Frequency') 
            plt.show()
            self.label_text_progress.set("Image Cumulated Histogram")

    def show_equalised_image(self):

        global matrixEqualised 


        #read image
        matrixEqualised, histoEqualised = histogram_equalization(matrix)
        photo = ImageTk.PhotoImage(image = Image.fromarray(matrixEqualised))
        #show image
        self.lbl_image.configure(image=photo)
        self.lbl_image.image=photo
        self.lbl_image.grid(padx=350)  
        self.label_text_progress.set("Image Equalised")

    def show_initial_image(self):
       
        #show image
        photo = ImageTk.PhotoImage(image = Image.fromarray(matrix))
        self.lbl_image.configure(image=photo)
        self.lbl_image.image=photo
        self.lbl_image.grid(padx=350)   
        self.label_text_progress.set("Initial Image")

    def show_histogram_equalised(self):
           
        matrixEqualised, histoEqualised = histogram_equalization(matrix)
        histoEq= histogram(matrixEqualised)
        plt.bar(range(256), histoEq)
        plt.xlabel('Graylevel / intensity')
        plt.ylabel('Frequency') 
        plt.show()
        self.label_text_progress.set("Equalised Histogarm")

    def showLinearTrans(self):
        matrixLT=linear_transformation(matrix)
        photo = ImageTk.PhotoImage(image = Image.fromarray(matrixLT))
        #show image
        self.lbl_image.configure(image=photo)
        self.lbl_image.image=photo
        self.lbl_image.grid(padx=350)  
        self.label_text_progress.set("Linear Transformation")

    def showSaturedTrans(self):
        matrixST = saturated_transformation(matrix,150,220)
        photo = ImageTk.PhotoImage(image = Image.fromarray(matrixST))
        #show image
        self.lbl_image.configure(image=photo)
        self.lbl_image.image=photo
        self.lbl_image.grid(padx=350)  
        self.label_text_progress.set("Satured Transformation")

    def showImageNoise(self):
        global matrixNoise
        matrixNoise = noise(matrix)
        photo = ImageTk.PhotoImage(image = Image.fromarray(matrixNoise))
        #show image
        self.lbl_image.configure(image=photo)
        self.lbl_image.image=photo
        self.lbl_image.grid(padx=350) 
        self.label_text_progress.set("Image with noise") 

    def showAverageFilter(self):
        matrixAverage=filer_average(matrixNoise)
        photo = ImageTk.PhotoImage(image = Image.fromarray(matrixAverage))
        #show image
        self.lbl_image.configure(image=photo)
        self.lbl_image.image=photo
        self.lbl_image.grid(padx=350) 
        self.label_text_progress.set("Average Filter for image with noise")
    
    def showMedianFilter(self):
        matrixMedian=filer_median(matrixNoise)
        photo = ImageTk.PhotoImage(image = Image.fromarray(matrixMedian))
        #show image
        self.lbl_image.configure(image=photo)
        self.lbl_image.image=photo
        self.lbl_image.grid(padx=350) 
        self.label_text_progress.set("Median Filter for image with noise")

    def showSNR_ratio(self):
        noiseMoyMatrix =filer_average(matrixNoise)
        noiseMedianMatrix =filer_median(matrixNoise)
        snrMoy=signal_to_Noise_Ratio(matrix,noiseMoyMatrix )
        snrMed=signal_to_Noise_Ratio(matrix,noiseMedianMatrix )
        self.label_text_progress.set("SNR ration for Average Filter: "+str(snrMoy)+"\n"+"SNR ration for median Filter: "+str(snrMed))

    def showThresh25(self):  
        matrix_thres25 = thresholding(matrix, 25)
        photo = ImageTk.PhotoImage(image = Image.fromarray(matrix_thres25))
        #show image
        self.lbl_image.configure(image=photo)
        self.lbl_image.image=photo
        self.lbl_image.grid(padx=350) 
        self.label_text_progress.set("Thersholding 25 limit")
        
    def showHistoThresh25(self): 
        matrix_thres25 = thresholding(matrix, 25)
        histo_thresh25 = histogram(matrix_thres25)
        plt.bar(range(256), histo_thresh25)
        plt.xlabel('Graylevel / intensity')
        plt.ylabel('Frequency') 
        plt.show()
        self.label_text_progress.set("Thersholding 25 limit Histogarm")

    def showThresh50(self):  
        matrix_thres50 = thresholding(matrix, 50)
        photo = ImageTk.PhotoImage(image = Image.fromarray(matrix_thres50))
        #show image
        self.lbl_image.configure(image=photo)
        self.lbl_image.image=photo
        self.lbl_image.grid(padx=350) 
        self.label_text_progress.set("Thersholding 50 limit")
        
    def showHistoThresh50(self): 
        matrix_thres50 = thresholding(matrix, 50)
        histo_thresh50 = histogram(matrix_thres50)
        plt.bar(range(256), histo_thresh50)
        plt.xlabel('Graylevel / intensity')
        plt.ylabel('Frequency') 
        plt.show()
        self.label_text_progress.set("Thersholding 50 limit Histogarm")

    def showThresh120(self):  
        matrix_thres120 = thresholding(matrix, 120)
        photo = ImageTk.PhotoImage(image = Image.fromarray(matrix_thres120))
        #show image
        self.lbl_image.configure(image=photo)
        self.lbl_image.image=photo
        self.lbl_image.grid(padx=350) 
        self.label_text_progress.set("Thersholding 120 limit")
        
    def showHistoThresh120(self): 
        matrix_thres120 = thresholding(matrix, 120)
        histo_thresh120 = histogram(matrix_thres120)
        plt.bar(range(256), histo_thresh120)
        plt.xlabel('Graylevel / intensity')
        plt.ylabel('Frequency') 
        plt.show()
        self.label_text_progress.set("Thersholding 120 limit Histogarm")

    def showThreshOtsu(self):
        matrix_image_otsu, maxSeuil = otsu(matrix)
        photo = ImageTk.PhotoImage(image = Image.fromarray(matrix_image_otsu))
        #show image
        self.lbl_image.configure(image=photo)
        self.lbl_image.image=photo
        self.lbl_image.grid(padx=350) 
        self.label_text_progress.set("Otsu Thersholding with limit "+str(maxSeuil))

    def showHistoThreshOtsu(self):
        matrix_image_otsu, maxSeuil = otsu(matrix)
        histo_threshOtsu = histogram(matrix_image_otsu)
        plt.bar(range(256), histo_threshOtsu)
        plt.xlabel('Graylevel / intensity')
        plt.ylabel('Frequency') 
        plt.show()
        self.label_text_progress.set("Otsu Thersholding "+str(maxSeuil)+" limit Histogarm")

    def showThreshMeanIntensity(self):
        mean, stdv = mean_stdev(matrix)
        matrix_mean=thresholding(matrix,mean)
        photo = ImageTk.PhotoImage(image = Image.fromarray(matrix_mean))
        #show image
        self.lbl_image.configure(image=photo)
        self.lbl_image.image=photo
        self.lbl_image.grid(padx=350) 
        self.label_text_progress.set("Mean Intensity Thersholding with mean value= "+str(mean))

    def showHistoThreshMeanIntensity(self):
        mean, stdv = mean_stdev(matrix)
        matrix_mean=thresholding(matrix,mean)
        histo_threshMean = histogram(matrix_mean)
        plt.bar(range(256), histo_threshMean)
        plt.xlabel('Graylevel / intensity')
        plt.ylabel('Frequency') 
        plt.show()
        self.label_text_progress.set("Mean Intensity Thersholding with mean value= "+str(mean))

    def showDilatation(self):
        image_otsu, maxSeuil = otsu(matrix)
        matrix_dil = dilatation(image_otsu)
        photo = ImageTk.PhotoImage(image = Image.fromarray(matrix_dil))
        #show image
        self.lbl_image.configure(image=photo)
        self.lbl_image.image=photo
        self.lbl_image.grid(padx=350) 
        self.label_text_progress.set("Dilatation")

    def showDilatationHisto(self):
        image_otsu, maxSeuil = otsu(matrix)
        matrix_dil = dilatation(image_otsu)
        histo_threshMean = histogram(matrix_dil)
        plt.bar(range(256), histo_threshMean)
        plt.xlabel('Graylevel / intensity')
        plt.ylabel('Frequency') 
        plt.show()
        self.label_text_progress.set("Dilatation Histogram")
 
    def showErrosion(self):
        image_otsu, maxSeuil = otsu(matrix)
        matrix_err = errosion(image_otsu)
        photo = ImageTk.PhotoImage(image = Image.fromarray(matrix_err))
        #show image
        self.lbl_image.configure(image=photo)
        self.lbl_image.image=photo
        self.lbl_image.grid(padx=350) 
        self.label_text_progress.set("Errosing")

    def showErrosionHisto(self):
        image_otsu, maxSeuil = otsu(matrix)
        matrix_err = errosion(image_otsu)
        histo_threshMean = histogram(matrix_err)
        plt.bar(range(256), histo_threshMean)
        plt.xlabel('Graylevel / intensity')
        plt.ylabel('Frequency') 
        plt.show()
        self.label_text_progress.set("Errosin Histogram")

    
    def showOpening(self):
        image_otsu, maxSeuil = otsu(matrix)
        matrix_opening = opening(image_otsu)
        photo = ImageTk.PhotoImage(image = Image.fromarray(matrix_opening))
        #show image
        self.lbl_image.configure(image=photo)
        self.lbl_image.image=photo
        self.lbl_image.grid(padx=350) 
        self.label_text_progress.set("Opening")

    def showOpeningHisto(self):
        image_otsu, maxSeuil = otsu(matrix)
        matrix_opening = opening(image_otsu)
        histo_threshMean = histogram(matrix_opening)
        plt.bar(range(256), histo_threshMean)
        plt.xlabel('Graylevel / intensity')
        plt.ylabel('Frequency') 
        plt.show()
        self.label_text_progress.set("Opening Histogram")
    
    def showClosing(self):
        image_otsu, maxSeuil = otsu(matrix)
        matrix_closing = closing(image_otsu)
        photo = ImageTk.PhotoImage(image = Image.fromarray(matrix_closing))
        #show image
        self.lbl_image.configure(image=photo)
        self.lbl_image.image=photo
        self.lbl_image.grid(padx=350) 
        self.label_text_progress.set("Closing")

    def showClosingHisto(self):
        image_otsu, maxSeuil = otsu(matrix)
        matrix_closing = closing(image_otsu)
        histo_threshMean = histogram(matrix_closing)
        plt.bar(range(256), histo_threshMean)
        plt.xlabel('Graylevel / intensity')
        plt.ylabel('Frequency') 
        plt.show()
        self.label_text_progress.set("Closing Histogram")
    
if __name__ == "__main__":
    app = Page()
    app.geometry("700x725+100+100")
    app.mainloop()

        
                                 
