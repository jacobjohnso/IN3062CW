#CreateSampleData--------------------------------------------------------------------
filename_read = os.path.join(path, "DataSet.csv") #this reads the DataSet file
data_frame = pd.read_csv(filename_read).fillna("c") #adds DataSet to the DataFrame and replaces NA characters
data_frame = data_frame.reindex(np.random.permutation(data_frame.index)) # randomises the data
print(data_frame)

mask = np.random.rand(len(data_frame)) < 0.01 #1% of dataset sample used(this is a very large data set around 200000rows of reviews)
trainDF = pd.DataFrame(data_frame[mask])
validationDF = pd.DataFrame(data_frame[~mask])

filename_write = os.path.join(path, "DataSet_Sample.csv") #creates a new csv file containing sample data
trainDF.to_csv(filename_write, index=True)
#------------------------------------------------------------------------------