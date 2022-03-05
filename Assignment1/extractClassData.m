function [databyclass,uql] = extractClassData(data,labels)
% EXTRACTCLASSDATA.M Extracts class data using the dataset and the class
% labels. 
% -Input is expected to have samples along the columns. Data dimensionality
% (features) should be on the increasing rows.
%The output is an indexable structure element of sample data
% indexed by the unique labels uql which is also an output.
uql = unique(labels);
    for i = 1:length(uql)
        idx = find(labels==uql(i));
        databyclass(uql(i)).samples = data(:,idx);
    end
    
end
