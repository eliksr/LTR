require('mobdebug').start()
--
-- Created by IntelliJ IDEA.
-- User: elik
-- Date: 1/15/17
-- Time: 12:32 PM
-- To change this template use File | Settings | File Templates.
--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'cunn'
require 'cudnn'

--require 'util.OneHot'
--require 'util.misc'

require 'image'
dofile 'util/improvider.lua'
require 'util.Csv'
require 'util.SpatialContrastiveNormalization__'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
-- cmd:argument('-model','model checkpoint to use for sampling')
cmd:option('-model','cv/iter0/lm_gru_goo_1024_lr1e-4_dr0.9_epoch33.02_5.0011.t7','model checkpoint to use for sampling')
cmd:option('-imfiles','/Users/elik/Documents/EliksDocs/thsis_files/learning-to-read-master/src/chestx/data/iter0_imcaps_trval_all_disease_only.csv','text file with input images and labels')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',1,' 0 to use max at each timestep, 1 to sample at each timestep')
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:option('-immodel','google','name of the imcnn model')
cmd:option('-immodelloc','/Users/elik/Documents/EliksDocs/thsis_files/learning-to-read-master/imcnn/logs/nin_bn_imagenet_lr0.1/model.net','path to the imcnn model')
cmd:option('-pcaloc','/Users/elik/Documents/EliksDocs/thsis_files/learning-to-read-master/src/chestx/data/iter0_imcaps_trval/pca_nin.t7','location of the pca file')
cmd:option('-data_loc','/Users/elik/Documents/EliksDocs/thsis_files/learning-to-read-master/data/chestx/ims_test','location of the image data')
cmd:text()

-- parse input params
opt = cmd:parse(arg)


function load_im_batches(imfiles, doaugs, dataloc, batchsize, imagesize, cropsize)
    local provider = Provider(imfiles, doaugs, dataloc, batchsize, imagesize, cropsize)
    provider:normalize()
    local ims = provider.trainData.data
    return ims
end

-- prepare image cnn model
local immodel = torch.load(opt.immodelloc)
--immodel:get(2):get(2):remove(#immodel:get(2):get(2).modules)
--immodel:get(2):get(2):remove(#immodel:get(2):get(2).modules)

local doaugs = {-1}
local batchsize = 1
local imsize = 256
local crsize = 224
local fnames = 'CXR660_IM-2237-0001-0001.png'

-- fnames = 'image.png',opt.data_loc = fullpath image dir,
local imbatch = load_im_batches(fnames, doaugs, opt.data_loc, batchsize, imsize, crsize)

local outputs = immodel:forward(imbatch)
print(outputs)

