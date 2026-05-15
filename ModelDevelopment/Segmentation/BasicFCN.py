import cv2
import inspect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import VolcDictionaryWithCorrectClears

data_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2/"
extra_channels_path = "C:/Users/ggp24ash/Documents/Scratch Data/CrossValidFoldSegmentation/InputChannels/woCotopaxi/"
sensor_mark_masks_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/SensorMarkMasks/"
target_masks_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/ProcessedLabels_UpdatedAfterReview/"

train_df = pd.read_excel("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/Dataset/DatasetSplits/UpdatedTVTSplits/CrossValidationSplits/CotopaxiLeftOut_Train.xlsx")
valid_df = pd.read_excel("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/Dataset/DatasetSplits/UpdatedTVTSplits/CrossValidationSplits/CotopaxiLeftOut_Valid.xlsx")
results_path = "RoughResults/Metrics.xlsx"
model_save_path = "RoughResults/Model.pth"
save_model = True
good_quality_only = True
mod = 1
train_batch_size = 10
valid_batch_size = 10
lr = 0.01
eps = 30

def show(image):
    plt.imshow(image, cmap="gray")
    plt.show()

def mask_sensor_marks(image, mask_path):
    if mask_path == "None":
        pass
    else:
        mask = cv2.imread(sensor_mark_masks_path + "/" + mask_path, -1)
        image = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
    return image
def read_data(df, mod):
    #[Observation index, channel index, 486, 648]
    X = []
    Y = []
    if good_quality_only == True:
        df = df[df["overall_obs"]=="No"]
        df.reset_index(inplace=True)
    for sample_index in range(0, df.shape[0], mod):
        image_name = df["image_name"][sample_index]
        dictionary_name = df["volcano_dictionary_name"][sample_index]
        dictionary = VolcDictionaryWithCorrectClears.map_dictionary_name_to_dictionary(dictionary_name)
        batch = df["labelling_batch_name"][sample_index]

        image_A = cv2.imread(data_path + image_name, -1)
        smmn_A = dictionary["sensor_marks_mask_A"]
        image_A = mask_sensor_marks(image_A, smmn_A)
        diff = np.load(extra_channels_path + "D_" + image_name[:-3] + ".npy")
        rel_AA = np.load(extra_channels_path + "A_" + image_name[:-3] + ".npy")

        observation = np.stack([image_A, diff, rel_AA], axis=0)

        #Read the target mask:
        mask_path = target_masks_path + batch + "/PlumeAndExpPixels_" + image_name.split(".")[0] + ".npy"
        two_channel_mask = np.load(mask_path)
        all_plume_mask = two_channel_mask[0, :, :] + two_channel_mask[1, :, :]
        all_plume_mask = np.where(all_plume_mask > 0, 1, 0)

        X.append(observation)
        Y.append(all_plume_mask)
    return X, Y

def min_max_scale_tensor(matrix, l, u):
    matrix = matrix - l
    new_max = u - l
    matrix = torch.divide(matrix, new_max)
    return matrix

class get_dataset(Dataset):
    '''Store dataset as torch tensors and set up method to load one pair.'''
    def __init__(self, dataframe, mod):
        print("Reading data:")
        x, y = read_data(dataframe, mod)
        print("Observations: " + str(len(x)))
        self.X = torch.tensor(np.array(x)).float()
        self.Y = torch.tensor(np.array(y)).float()
    def calculate_minmax_params(self):
        #Calculate the parameters needed to scale the inputs to range [0, 1]
        print("Calculating min and max params.")
        mins_x = []
        maxs_x = []
        for c in range(0, self.X.shape[1]):
            mins_x.append(torch.min(self.X[:,c,:,:]).item()) #Keep the channel dimension
            maxs_x.append(torch.max(self.X[:,c,:,:]).item())

        return (mins_x, maxs_x)
    def calculate_m_sd_params(self):
        '''For each channel, calculate the mean and standard devitation.'''
        print("Calculating mean and std params.")
        means_x = []
        stds_x = []
        for c in range(0, self.X.shape[1]):
            means_x.append(torch.mean(self.X[:,c,:,:]).item())
            stds_x.append(torch.std(self.X[:,c,:,:]).item())
        return (means_x, stds_x)
    def min_max_scale_inputs(self, params):
        for channel in range(0, self.X.shape[1]):
            self.X[:,channel,:,:] = min_max_scale_tensor(self.X[:,channel,:,:], params[0][channel], params[1][channel])
    def normalise_inputs(self, params):
        for channel in range(0, self.X.shape[1]):
            self.X[:,channel,:,:] = self.X[:,channel,:,:] - params[0][channel]
            self.X[:, channel, :, :] = torch.divide(self.X[:,channel,:,:], params[1][channel])

    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        return self.X[index].to(device), self.Y[index].to(device)

class FCN(torch.nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.res = torchvision.models.segmentation.fcn_resnet50(weights='COCO_WITH_VOC_LABELS_V1')

        for param in self.res.parameters():
            param.requires_grad = False

        self.res.classifier[4] = torch.nn.Conv2d(512, 1, kernel_size=(1,1), stride=(1,1))

        for param in self.res.classifier.parameters():
            param.requires_grad = True

        print(self.res)
        summary(self.res)
        print(inspect.getsource(self.res.forward))
        self.res.classifier.add_module("sigmoid", torch.nn.Sigmoid())
        #TODO note I've not added sigmoid to the aux classifier

    def forward(self, x):
        return self.res(x)


def IOU(predicted, true, threshold):
    i = torch.where(torch.logical_and(predicted >= threshold, true==1), 1, 0)
    u = torch.where(torch.logical_or(predicted >= threshold, true==1), 1, 0)
    #Want to sum over last two dimensions (width and height)
    #[Obs index, h, w]
    I = torch.sum(i, dim=(-1, -2))
    U = torch.sum(u, dim=(-1, -2))

    result = torch.where(U!=0, torch.divide(I, U), 0)
    #TODO what if there is no plume in the image, this is currently giving value of zero
    return result


def train_batch(model, data, loss_fn, optimizer):
    model.train()
    inputs, targets = data
    prediction = model(inputs)["out"][:,0,:,:]
    optimizer.zero_grad()
    loss = loss_fn(prediction, targets) #Loss for every pixel of every image
    sample_mean_loss = torch.mean(loss, dim=(-1, -2)) #Mean loss (over all pixels) for each sample
    batch_mean_loss = torch.mean(sample_mean_loss) #Overall mean loss for the batch
    acc = IOU(prediction, targets, threshold=0.5)
    batch_mean_loss.backward()
    optimizer.step()
    return sample_mean_loss, acc

@torch.no_grad()
def validate_batch(model, val_data, loss_fn):
    model.eval()
    inputs, targets = val_data
    predictions = model(inputs)["out"][:,0,:,:]
    loss = loss_fn(predictions, targets) #Calculate the loss for every pixel of every image
    sample_mean_loss = torch.mean(loss, dim=(-1, -2))  # Mean loss (over all pixels) for each sample
    acc = IOU(predictions, targets, threshold=0.5)
    return sample_mean_loss, acc

def plot_metrics(df):
    ep_indexes = np.arange(1, df.shape[0] + 1, 1)
    fig, axs = plt.subplots(ncols=2)
    ta_plot = axs[0].plot(ep_indexes, df["train_IOU"].tolist(), label="Train")
    va_plot = axs[0].plot(ep_indexes, df["valid_IOU"].tolist(), label="Valid")
    tl_plot = axs[1].plot(ep_indexes, df["train_loss"].tolist(), label="Train")
    vl_plot = axs[1].plot(ep_indexes, df["valid_loss"].tolist(), label="Valid")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("IOU (Mean of all samples)")
    axs[0].legend()
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Mean BCE Loss (Mean over samples)")
    axs[1].legend()
    plt.suptitle("Training Metrics: ")
    plt.savefig(results_path[:-4] + "png")
    plt.show()


def apply_and_visualise(model, dataloader, dataframe):
    model.eval()
    for vi, sample in enumerate(dataloader):
        image_name = dataframe["image_name"][vi]
        intake, target = sample
        prediction = model(intake)["out"][0,0,:,:]
        prediction_to_plot = prediction.detach().cpu().numpy()
        target_to_plot = target[0,:,:].detach().cpu().numpy()
        original_to_plot = intake[0,0,:,:].detach().cpu().numpy()
        fig, axs = plt.subplots(ncols=3)
        axs[0].imshow(original_to_plot, cmap="gray")
        axs[1].imshow(prediction_to_plot, cmap="gray")
        axs[2].imshow(target_to_plot)
        plt.show()






device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:" + device)

train_set = get_dataset(train_df, mod)
#Scale the train set to [0,1] with min-max scaling, and then normalise by
#mean and sd. Save the params and use these to apply the same transforms
#to the validation set.
mm_params = train_set.calculate_minmax_params()
train_set.min_max_scale_inputs(mm_params)
msd_params = train_set.calculate_m_sd_params()
train_set.normalise_inputs(msd_params)
valid_set = get_dataset(valid_df, mod)
valid_set.min_max_scale_inputs(mm_params)
valid_set.normalise_inputs(msd_params)

train_dataloader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, drop_last=False)
valid_dataloader = DataLoader(valid_set, batch_size=valid_batch_size, shuffle=True, drop_last=False)


model = FCN().to(device)
loss_fn = torch.nn.BCELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
results_df = pd.DataFrame(columns=["epoch", "train_loss", "train_IOU", "valid_loss", "valid_IOU"] )
for e in range(1, eps + 1):
    print("Epoch " + str(e))
    etl = []
    eta = []
    evl = []
    eva = []
    for t_batch, t_data in enumerate(train_dataloader):
        sample_mean_train_loss, train_acc = train_batch(model, t_data, loss_fn, optimizer)
        etl.extend(sample_mean_train_loss.detach().cpu().numpy().tolist())
        eta.extend(train_acc.detach().cpu().numpy().tolist())
    for val_batch, val_data in enumerate(valid_dataloader):
        sample_mean_val_loss, val_acc = validate_batch(model, val_data, loss_fn)
        evl.extend(sample_mean_val_loss.detach().cpu().numpy().tolist())
        eva.extend(val_acc.detach().cpu().numpy().tolist())

    tl = np.mean(np.array(etl))
    ta = np.mean(np.array(eta))
    vl = np.mean(np.array(evl))
    va = np.mean(np.array(eva))
    results_df.loc[len(results_df)] = [e, tl, ta, vl, va]

results_df.to_excel(results_path)
plot_metrics(results_df)
if save_model == True:
    torch.save(model.state_dict(), model_save_path)



####Apply saved model and visualise on validation set

model = FCN().to(device)
model.load_state_dict(torch.load(model_save_path, weights_only=True))
vis_mod=5
vis_dataloader = DataLoader(valid_set, batch_size=1, shuffle=False, drop_last=False)
apply_and_visualise(model, vis_dataloader, valid_df)

