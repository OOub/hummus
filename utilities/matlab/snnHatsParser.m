% snnHatsParser.m

% Created by Omar Oubari 
% PhD - Institut de la Vision
% Email: omar.oubari@inserm.fr

% Last Version: 04/10/2018

% Information: snnHatsParser is a function that generates Histograms of averaged time surfaces (HATS) 
% from the scaled 64x56 n-Cars database and parses it so it can be fed into the Adonis spiking neural network simulator

% Dependencies: hats.m - load_atis_data.m

function [output, labels, gridH] = snnHatsParser(pathToCars, baseFileNames, samplePercentage, r, tau, dt, patternDuration, repetitions, timeJitter, timeBetweenPresentations, boolRandomisePresentationOrder, pathToBackgrounds, save)
    % folderPath - the path to the folder where all the recordings we want to parse are located

    % baseFileNames - the common name between all the files we want to feed
    % into the SNN so the dir method can locate them inside the folder
    % specified by the folderPath

    % samplePercentage (optional) - randomly select a percentage of the
    % files present in the data folder, in case we only want to train with
    % a sample, and not the full data. 
    
    % r (optional) - radius
    
    % tau (optional) - decay of the time surface
    
    % dt (optional) - temporal window of the local memory time surface
    
    % patternDuration (optional) - total duration of the patterns
    
    % repetitions (optional) - number of times each recording is presented

    % timeJitter (optional) - adds time jitter to the recording. The value
    % is the standard deviation for the gaussian centered around a spike time
    
    % timeBetweenPresentations (optional) - time in microseconds between each
    % repetition
    
    % boolRandomisePresentationOrder (optional) - bool to select whether to
    % randomise the order of appeance of the recordings

    % pathToBackgrounds (optional) - in case we want to add the background
    % to the data, for testing purposes
    
    % save (optional) - true to save the files, false otherwise
    
    % handling optional arguments
    if nargin < 3
        samplePercentage = 100;
        r = 3;
        tau = 1e9;
        dt = 1e5;
        patternDuration = 100;
        repetitions = 1;
        timeJitter = 0;
        timeBetweenPresentations = 100;
        boolRandomisePresentationOrder = false;
        pathToBackgrounds = '';
        save = true;
    elseif nargin < 4
        r = 3;
        tau = 1e9;
        dt = 1e5;
        patternDuration = 100;
        repetitions = 1;
        timeJitter = 0;
        timeBetweenPresentations = 100;
        boolRandomisePresentationOrder = false;
        pathToBackgrounds = '';
        save = true;
    elseif nargin < 5
        tau = 1e9;
        dt = 1e5;
        patternDuration = 100;
        repetitions = 1;
        timeJitter = 0;
        timeBetweenPresentations = 100;
        boolRandomisePresentationOrder = false;
        pathToBackgrounds = '';
        save = true;
    elseif nargin < 6
        dt = 1e5;
        patternDuration = 100;
        repetitions = 1;
        timeJitter = 0;
        timeBetweenPresentations = 100;
        boolRandomisePresentationOrder = false;
        pathToBackgrounds = '';
        save = true;
    elseif nargin < 7
        patternDuration = 100;
        repetitions = 1;
        timeJitter = 0;
        timeBetweenPresentations = 100;
        boolRandomisePresentationOrder = false;
        pathToBackgrounds = '';
        save = true;
    elseif nargin < 8
        repetitions = 1;
        timeJitter = 0;
        timeBetweenPresentations = 100;
        boolRandomisePresentationOrder = false;
        pathToBackgrounds = '';
        save = true;
    elseif nargin < 9
        timeJitter = 0;
        timeBetweenPresentations = 100;
        boolRandomisePresentationOrder = false;
        pathToBackgrounds = '';
        save = true;
    elseif nargin < 10
        timeBetweenPresentations = 100;
        boolRandomisePresentationOrder = false;
        pathToBackgrounds = '';
        save = true;
    elseif nargin < 11
        boolRandomisePresentationOrder = false;
        pathToBackgrounds = '';
        save = true;
    elseif nargin < 12
        pathToBackgrounds = '';
        save = true;
    elseif nargin < 13
        save = true;
    end
    
    % searching for all files fitting the description specified by the
    % folderPath and the baseFileNames
    datasetDirectory = dir(strcat(pathToCars,'*',baseFileNames, '*'));
    
    % take a sample from the folder
    if samplePercentage < 100
        numberOfSamples = floor((length(datasetDirectory)*samplePercentage)/100);
        datasetDirectory = datasample(datasetDirectory, numberOfSamples);
    end
    
    labels = [];
    % add test data
    if ~isempty(pathToBackgrounds)
        testDirectory = dir(strcat(pathToBackgrounds,'*',baseFileNames, '*'));
        if samplePercentage < 100
            numberOfSamples = floor((length(testDirectory)*samplePercentage)/100);
            testDirectory = datasample(testDirectory, numberOfSamples);
        end
%         cars = ones(length(datasetDirectory),1);
%         backgrounds = zeros(length(testDirectory),1);
        
        cars = cellstr(repmat('car',[length(datasetDirectory) 1]));
        backgrounds = cellstr(repmat('bgd', [length(testDirectory) 1]));
        
        datasetDirectory = [datasetDirectory;testDirectory];
        labels = [cars;backgrounds];
    else 
        cars = cellstr(repmat('car',[length(datasetDirectory) 1]));
        labels = cars;
    end
    
    % convert data from TD to spikes
    H = {}; gridH = {}; spikeH = {};
    disp('files being parsed:')
    for i = 1:length(datasetDirectory)
        disp(strcat(datasetDirectory(i).folder, '/', datasetDirectory(i).name));
        data = load_atis_data(strcat(datasetDirectory(i).folder, '/', datasetDirectory(i).name));
        [H{i,1},gridH{i,1}] = hats([data.x, data.y, data.ts, data.p], r, tau, dt);
        scaledH{i,1} = zeros(size(gridH{i,1},1), size(gridH{i,1},2));
        
        temp = [];
        for j = 1:size(gridH{i,1},1)
            for k = 1:size(gridH{i,1},2)
                % only making the active regions spike
                scaledH{i,1}(j,k) = (((gridH{i,1}(j,k) - min(gridH{i,1}(:))) * 255) / (max(gridH{i,1}(:)) - min(gridH{i,1}(:)))) / 6;
                if scaledH{i,1}(j,k) > 0
                    % Poisson encoding
                    spikeTrain = poissonSpikeGenerator(scaledH{i,1}(j,k), patternDuration);
                    temp(end+1:end+length(spikeTrain), :) = [spikeTrain, repmat(j-1, [length(spikeTrain) 1]), repmat(k-1, [length(spikeTrain) 1])];
                end
            end
        end
        spikeH{i,1} = sortrows(temp,1);
    end
    
    % generating the SNN input
    presentationOrder = repmat([1:length(spikeH)]',[repetitions 1]);
    labels = repmat(labels, [repetitions 1]);
    
    if boolRandomisePresentationOrder == true
        presentationOrder = presentationOrder(randperm(size(presentationOrder,1)),:);
        labels = labels(presentationOrder);
    end
    
    snnInput = []; spikeIntervals = [];
    for i = 1:length(presentationOrder)
        snnInput = [double(snnInput); double(spikeH{presentationOrder(i)}(:,1)), double(spikeH{presentationOrder(i)}(:,2)), double(spikeH{presentationOrder(i)}(:,3))];
        spikeIntervals(end+1,:) = length(snnInput(1:end,1));
    end

    % Shifting the timestamps so the presentations are sequential
    firstRowData = []; index = [];
    for i = 1:length(spikeH)
        firstRowData(end+1,:) = [spikeH{i}(1,1), spikeH{i}(1,2), spikeH{i}(1,3)];
        temp = find(snnInput(:,1) == firstRowData(end,1) & snnInput(:,2) == firstRowData(end,2) & snnInput(:,3) == firstRowData(end,3));
        index = [index;temp];
    end
    index = sort(index);
    index(end+1) = length(snnInput)+1;
    
    index2 = index(1);
    for i = 2:length(index)
        if index(i) - index(i-1) > 1
            index2(end+1,:) = index(i);
        end
    end 
    
    for i = 3:size(index2,1)
        snnInput(index2(i-1):index2(i)-1,1) = snnInput(index2(i-1):index2(i)-1,1) + snnInput(index2(i-1)-1) + timeBetweenPresentations;
    end
    
    % adding time jitter
    if timeJitter > 0
        for i = 1:size(snnInput,1)
            jitter = normrnd(snnInput(i,1), timeJitter);
            while (jitter < 0)
                jitter = normrnd(snnInput(i,1), timeJitter);
            end
            snnInput(i,1) = jitter;
        end
        snnInput = sortrows(snnInput,1);
    end

    labelTimes = num2cell(snnInput(1, 1));
    labelTimes = [labelTimes;num2cell(snnInput(spikeIntervals(1:end-1)+1, 1))];

    output = struct('snnInput',snnInput,'spikeIntervals',spikeIntervals,'presentationOrder', presentationOrder);
    labels = [labels,labelTimes];
    
    if save == true
        if ~isempty(pathToBackgrounds)
            databaseType = 'test_';
        else
            databaseType = 'train_';
        end
        filename = strcat('nCars_', databaseType, num2str(samplePercentage), 'samplePerc_', num2str(repetitions),'rep');
        dlmwrite(strcat(filename,'.txt'), snnInput, 'delimiter', ' ', 'precision', '%f');
        labelWriter(strcat(filename,'Label.txt'), labels);
    end
end

function [spikeTime] = poissonSpikeGenerator(fr, tSim)
    tSim = tSim*1e-3;
    dt = 1e-3;
    nBins = floor(tSim/dt);
    spikeMat = rand(1, nBins) < fr*dt;
    tVec = 0:dt:tSim-dt;
    spikeTime = find(spikeMat == 1)';
end

function labelWriter(filename, labels)
    fid = fopen(filename,'wt');
    for row = 1:size(labels,1)
        fprintf(fid,labels{row,1});
        fprintf(fid, ' ');
        fprintf(fid, num2str(labels{row,2}));
        fprintf(fid, '\n');
    end
    fclose(fid);
end