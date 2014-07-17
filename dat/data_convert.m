function [] = data_convert( filename, X, type )

narginchk(3, 3);

fid = fopen(filename, 'w');
if fid == -1
    error(['error opening the file: ', filename]);
end
fwrite(fid, size(X), 'int');
fwrite(fid, X, type);
fclose(fid);

end

% function [] = convert( fid, X, Y )
% 
% for i = 1 : size(X, 2)
%     if isempty(Y)
%         fprintf(fid, '0');
%     else
%         fprintf(fid, '%d', Y(i));
%     end
%     for j = find(X(:, i))'
%         fprintf(fid, ' %d:%.16g', j, X(j, i));
%     end
%     fprintf(fid, '\n');
% end
% 
% end

