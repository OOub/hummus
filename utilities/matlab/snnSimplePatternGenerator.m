clc
clear
close all

addpath('../src');

%% FEATURES
timeJitter = 1; % 0 to turn off time jitter and 1 to add time jitter
additiveNoise = 0; % percentage of background noise
neuronalNoise = 0; % percentage of neural noise
randomizePatternOccurence = 0; % 0 to turn off and 1 to randomize

%% PARAMETERS
numberOfPatterns = 4;
repetitions = 20000;
timeBetweenIntervals = 10;
timejitterInterval = [0,1.5]; %standard deviation for the jitter

%% PATTERN CREATION
pattern1 = [0;1;2;3];
time1 = [5;10;12;24];

pattern2 = [4;5;6;7];
time2 = [1;4;5;10];

pattern3 = [10;18;1;20];
time3 = [3;5;8;20];

pattern4 = [22;26;25;23];
time4 = [5;10;15;20];

patternsCollections = struct('patterns',horzcat(pattern1,pattern2,pattern3,pattern4),'times',horzcat(time1,time2,time3,time4));
numberOfSpikesInPattern = size(patternsCollections.patterns,1);

patternIndices = repmat([1:numberOfPatterns]',repetitions+1,1);
if (randomizePatternOccurence == 1)
    patternIndices = patternIndices(randperm(size(patternIndices, 1)), :);
end

for i = 1:length(patternIndices)
    data(:,i) = patternsCollections.patterns(:,patternIndices(i));
    time(:,i) = patternsCollections.times(:,patternIndices(i));
end

data = reshape(data,size(data,1)*size(data,2),1);
time = reshape(time,size(time,1)*size(time,2),1);

shift = 0;
for i = 1:length(time)
    if (timeJitter == 1)
        time(i) = time(i) + (timejitterInterval(1) + (timejitterInterval(2) - timejitterInterval(1)).*rand(1,1)) + shift; % time jitter
    elseif (timeJitter == 0)
        time(i) = time(i) + shift; % no time jitter
    end
    
    if mod(i, numberOfSpikesInPattern) == 0
        shift = time(i) + timeBetweenIntervals;
    end
end
clear i pattern1 pattern2 pattern3 pattern4 time1 time2 time3 time4 shift

%% ADDING NEURONAL NOISE
numberNeuronalNoise = floor(length(data)*neuronalNoise);
if (numberNeuronalNoise > 0)
    replacementIndex = randi(length(data),numberNeuronalNoise,1);
    for i = 1:length(replacementIndex)
        data(replacementIndex(i)) = randi(max(data));
    end
end

%% ADDING BACKGROUND NOISE
numberNoiseInsertion = floor(length(data)*additiveNoise);
if (numberNoiseInsertion > 0)
    insertionIndex = randi(length(data),numberNoiseInsertion,1);
    for i = 1:length(insertionIndex)
        neuronNoise = randi(max(data));
        tmin = time(insertionIndex(i));
        tmax = time(insertionIndex(i)+1);
        timeNoise=tmin+rand(1,1)*(tmax-tmin);
        data = insertrows(data,neuronNoise,insertionIndex(i));
        time = insertrows(time,timeNoise,insertionIndex(i));
    end
end

%% SAVING TO TEXT FILE
data_matrix = [time,data];
if (timeJitter == 1)
    if (randomizePatternOccurence == 1)
        dlmwrite(strcat("random_",num2str(timejitterInterval(2)),"timeJitter",num2str(additiveNoise),"bn",num2str(neuronalNoise),"nn",num2str(numberOfPatterns),'fakePatterns_snnTest_',num2str(repetitions),'reps_',num2str(timeBetweenIntervals),'msInterval.txt'), data_matrix, 'delimiter', ' ','precision','%f')
    else
        dlmwrite(strcat(num2str(timejitterInterval(2)),"timeJitter",num2str(additiveNoise),"bn",num2str(neuronalNoise),"nn",num2str(numberOfPatterns),'fakePatterns_snnTest_',num2str(repetitions),'reps_',num2str(timeBetweenIntervals),'msInterval.txt'), data_matrix, 'delimiter', ' ','precision','%f')
    end
elseif (timeJitter == 0)
    if (randomizePatternOccurence == 1)
        dlmwrite(strcat("random_",num2str(additiveNoise),"bn",num2str(neuronalNoise),"nn",num2str(numberOfPatterns),'fakePatterns_snnTest_',num2str(repetitions),'reps_',num2str(timeBetweenIntervals),'msInterval.txt'), data_matrix, 'delimiter', ' ','precision','%f')
    else
        dlmwrite(strcat(num2str(additiveNoise),"bn",num2str(neuronalNoise),"nn",num2str(numberOfPatterns),'fakePatterns_snnTest_',num2str(repetitions),'reps_',num2str(timeBetweenIntervals),'msInterval.txt'), data_matrix, 'delimiter', ' ','precision','%f')
    end
end