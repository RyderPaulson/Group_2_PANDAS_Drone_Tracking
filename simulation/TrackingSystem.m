classdef TrackingSystem < matlab.System
    % TrackingSystem System object bridge to Python tracking system
    %
    % This system processes video frames from Simulink through the Python
    % tracking pipeline and returns annotated frames with tracked coordinates.
    
    properties (Nontunable)
        TextPrompt = 'drone'           % Object detection prompt
        WindowSize = 16                 % CoTracker window size
        RstIntervalMult = 16           % Reset interval multiplier
        BbCheckMult = 8                % Bounding box check multiplier
        MaxImgSize = 800               % Maximum image dimension
    end
    
    properties (Access = private)
        PyBridge                        % Python bridge object
        FrameHeight              % Input frame height
        FrameWidth               % Input frame width
        IsInitialized = false           % Initialization flag
    end
    
    methods (Access = protected)
        function setupImpl(obj, first_frame)
            % Initialize the Python bridge and tracking system
            
            % Get frame dimensions from first input
            [obj.FrameHeight, obj.FrameWidth, ~] = size(first_frame);
            
            if isempty(coder.target)
                % Add Python path (parent directory from simulation folder)
                current_path = pwd;
                [~, current_folder, ~] = fileparts(current_path);

                if strcmp(current_folder, 'simulation')
                    cd ..;
                    project_folder = pwd;

                    % Import and initialize Python bridge
                    py.importlib.import_module('SimBridge');
                    obj.PyBridge = py.SimBridge.TrackingBridge(...
                        pyargs(...
                            'text_prompt', obj.TextPrompt, ...
                            'window_size', int32(obj.WindowSize), ...
                            'rst_interval_mult', int32(obj.RstIntervalMult), ...
                            'bb_check_mult', int32(obj.BbCheckMult), ...
                            'max_img_size', int32(obj.MaxImgSize) ...
                        ) ...
                    );

                    cd simulation;
                end
            end
            
            obj.IsInitialized = true;
        end
        
        function out_frame = stepImpl(obj, src_frame)
            % Process frame through Python tracking system
            %
            % Inputs:
            %   src_frame - Input frame [H x W x 3] uint8 RGB image
            %
            % Outputs:
            %   anno_frame - Annotated frame [H x W x 3] uint8 RGB image
            
            frame_py = obj.matlabToPython(src_frame);
            annotated_frame_py = obj.PyBridge.process_frame(frame_py);
            out_frame = obj.pythonToMatlab(annotated_frame_py);
        end
    
        function num = getNumOutputsImpl(~)
            num = 1;
        end
        
        function out = getOutputDataTypeImpl(~)
            out = 'uint8';
        end

        function icon = getIconImpl(~)
            % Fun little function to add an icon to the block.
            icon = matlab.system.display.Icon('.\Panda-Logo.jpg');
        end
    end
    
    methods (Access = private)
        function py_array = matlabToPython(obj, matlab_img)
            % Convert MATLAB uint8 image to Python numpy array
            %
            % Input: [H x W x 3] uint8 MATLAB array
            % Output: Python numpy array in correct format for OpenCV
            
            % Reshape to column vector for efficient transfer
            img_vec = reshape(matlab_img, [], 1);
            
            % Create numpy array from MATLAB data
            np = py.importlib.import_module('numpy');
            py_array = np.array(img_vec, dtype=np.uint8);
            
            % Reshape to original image dimensions
            py_array = py_array.reshape(int32(obj.FrameHeight), ...
                                       int32(obj.FrameWidth), ...
                                       int32(3));
        end
        
        function matlab_img = pythonToMatlab(obj, py_array)
            % Convert Python numpy array back to MATLAB format
            %
            % Input: Python numpy array [H x W x 3]
            % Output: MATLAB uint8 array [H x W x 3]
            
            % Convert to MATLAB array
            matlab_vec = uint8(py_array.flatten().tolist());
            
            % Reshape to image dimensions
            matlab_img = reshape(matlab_vec, obj.FrameHeight, obj.FrameWidth, 3);
        end
    end
end