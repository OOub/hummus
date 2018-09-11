% snnAtisDataParser.m

% Created by Omar Oubari 
% PhD - Institut de la Vision
% Email: omar.oubari@inserm.fr

% Last Version: 04/09/2018

% Information: snnAtisDataParser is a function that parses recordings originating from the Atis cameras, so it can
% be fed into the Adonis spiking neural network simulator. The recordings
% can be presented multiple times in a randomised or sequential order. 

function [output, recordings] = snnAtisDataParser(folderPath, baseFileNames, repetitions, timeBetweenPresentations, timeJitter, conversionFactor, boolRandomisePresentationOrder, boolSpatialCrop, boolTemporalCrop)
    % folderPath - the path to the folder where all the recordings we want to parse are located

    % baseFileNames - the common name between all the files we want to feed
    % into the SNN so the dir method can locate them inside the folder
    % specified by the folderPath
    
    % repetitions - number of times each recording is presented
    
    % timeBetweenPresentations - time in microseconds between each
    % repetition
    
    % timeJitter (optional) - adds time jitter to the recording. The value
    % is the standard deviation for the gaussian centered around a spike time
    
    % conversionFactor (optional) - to convert data from microseconds
    
    % boolRandomisePresentationOrder (optional) - bool to select whether to
    % randomise the order of appeance of the recordings
    
    % boolSpatialCrop (optional) - bool to select if we want to spatially crop the recordings

    % boolTemporalCrop (optional) - bool to select if we want to temporally crop the recordings
    
    
    % handling optional arguments
    if nargin < 5
        timeJitter = 0;
        conversionFactor = 10^-3;
        boolRandomisePresentationOrder = false;
        boolSpatialCrop = false;
        boolTemporalCrop = false; 
    elseif nargin < 6
        conversionFactor = 10^-3;
        boolRandomisePresentationOrder = false;
        boolSpatialCrop = false;
        boolTemporalCrop = false; 
    elseif nargin < 7
        boolRandomisePresentationOrder = false;
        boolSpatialCrop = false;
        boolTemporalCrop = false;  
    elseif nargin < 8
        boolSpatialCrop = false;
        boolTemporalCrop = false;
    elseif nargin < 9
        boolTemporalCrop = false;
    end

    % searching for all files fitting the description specified by the
    % folderPath and the baseFileNames
    datasetDirectory = dir(strcat(folderPath,'*',baseFileNames, '*'));

    disp('files being parsed:')
    for i = 1:length(datasetDirectory)
        disp(strcat(datasetDirectory(i).folder, '/', datasetDirectory(i).name));
        data{i,1} = load_atis_data(strcat(datasetDirectory(i).folder, '/', datasetDirectory(i).name));
        parsedData{i,1} = data{i,1};
        
        % temporal crop
        if boolTemporalCrop == true
            prompt = {'Start Timestamp:','End Timestamp:'};
            title = 'Temporal Crop';
            dims = [1 35];
            definput = {'5000','10000'};
            answer = inputdlg(prompt,title,dims,definput);

            parsedData{i,1} = temporalCrop(parsedData{i,1}, str2double(answer{1}), str2double(answer{2}));
            
            % reset timestamp to zero
            parsedData{i,1}.ts = parsedData{i,1}.ts - parsedData{i,1}.ts(1);
        end
        
        %spatial crop
        if boolSpatialCrop == true
            prompt = {'Left:','Bottom:','Width:','Height'};
            title = 'Spatial Crop';
            dims = [1 35];
            definput = {'100','120','90','90'};
            answer = inputdlg(prompt,title,dims,definput);

            parsedData{i,1} = spatialCrop(parsedData{i,1}, str2double(answer{1}), str2double(answer{2}), str2double(answer{3}), str2double(answer{4}));
        end
    end

    % generating the SNN input
    presentationOrder = repmat([1:length(parsedData)]',[repetitions 1]);

    if boolRandomisePresentationOrder == true
        presentationOrder = Shuffle(presentationOrder);
    end
    
    snnInput = []; spikeIntervals = [];
    for i = 1:length(presentationOrder)
        snnInput = [double(snnInput); double(parsedData{presentationOrder(i)}.ts), double(parsedData{presentationOrder(i)}.x), double(parsedData{presentationOrder(i)}.y)];
        spikeIntervals(end+1,:) = length(snnInput(1:end,1));
    end
    
    % Shifting the Timestamps so the presentations are sequential
    firstRowData = []; index = [];
    for i = 1:length(parsedData)
        firstRowData(end+1,:) = [parsedData{i}.ts(1), parsedData{i}.x(1), parsedData{i}.y(1)];
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
    end
    
    % converting from microseconds
    snnInput(:,1) = snnInput(:,1)*conversionFactor;
    
    for i = 1:size(spikeIntervals)
        presentationOrder(i,3) = snnInput(spikeIntervals(i),1);
    end

    output = struct('snnInput',snnInput,'spikeIntervals',spikeIntervals);
    recordings = struct('originalData',data,'parsedData',parsedData);
end

function [output] = spatialCrop(filename, left, bottom, width, height)
    % only keeps events that occur within a spatial rectangle (left+width) * (bottom+height)
    mask = and(and(filename.x>=left, filename.x<(left+width)), and(filename.y>=bottom, filename.y<(bottom+height)));
    fields = fieldnames(filename);
    for i = 1:numel(fields)
        output.(fields{i}) = filename.(fields{i})(mask);
    end
end

function [output] = temporalCrop(filename, ts1, ts2)
    % only keeps events that have timestamps within [ts1, ts2]
    mask = and(filename.ts >= ts1, filename.ts <= ts2);
    fields = fieldnames(filename);
    for i = 1:numel(fields)
        output.(fields{i}) = filename.(fields{i})(mask);
    end
end