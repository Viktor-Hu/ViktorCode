pcapFileName = 'Truck.pcap';
calibFileName = 'Truck.json';
ousterReader = ousterFileReader(pcapFileName,calibFileName);

ousterReader.CurrentTime = ousterReader.StartTime + seconds(0.3);
frame = 0
while(hasFrame(ousterReader))
    ptCloud = readFrame(ousterReader);
    data_x = ptCloud.Location(:,:,1);
    data_y = ptCloud.Location(:,:,2);
    data_z = ptCloud.Location(:,:,3);
    data_i = double(ptCloud.Intensity);
    % 转为1列
    point_x = data_x(:);
    point_y = data_y(:);
    point_z = data_z(:);
    point_i = data_i(:)';
    Mapped = mapminmax(point_i, 0, 1); % 将反射强度归一化
    point_i = Mapped'; % 归一化后在转为1列
    ALL_data = [point_x, point_y, point_z, point_i];
    new_data = [];
    for i = 1: length(ALL_data)
        if ALL_data(i,4) ~= 0
            new_data = [new_data, ALL_data(i,:)];
        end
    end
    if frame<10
        fid=fopen(strcat('E:\Study\jason\sfa3d\newdata\output\00000',num2str(frame),'.bin'),'w');
    elseif (10 <= frame) && (frame < 100)
        fid=fopen(strcat('E:\Study\jason\sfa3d\newdata\output\0000',num2str(frame),'.bin'),'w');
    elseif (100 <= frame) && (frame < 1000)
        fid=fopen(strcat('E:\Study\jason\sfa3d\newdata\output\000',num2str(frame),'.bin'),'w');
    elseif (1000 <= frame) && (frame < 10000)
        fid=fopen(strcat('E:\Study\jason\sfa3d\newdata\output\00',num2str(frame),'.bin'),'w');
    end

    fwrite(fid,new_data, "float32");
    fclose(fid);
    frame = frame+1
end