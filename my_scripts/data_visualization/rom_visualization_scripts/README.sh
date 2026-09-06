# MHR ROM tools

This folder contains the MHR range-of-motion (ROM) calculations, JSON-to-MHR
reconstruction utilities, and the MHR ROM visualizers.

## Files

- `mhr_shoulder.py` — shoulder flexion, abduction, and internal/external rotation.
- `mhr_hip.py` — hip flexion, extension, abduction, and internal/external rotation.
- `mhr_json.py` — reconstructs MHR mesh and joints from Synthium JSON pose parameters.
- `visualize_mhr_json_rom.py` — recommended entry point for Synthium MHR JSON files.
- `visualize_mhr_shoulder_flexion.py` — NPZ-based shoulder visualizer.
- `visualize_mhr_hip.py` — NPZ-based hip flexion/extension/abduction visualizer.
- `visualize_mhr_hip_rotation.py` — NPZ-based hip rotation visualizer.

The JSON commands below use the original video timeline. The selected frame is
`round(time * fps)`. Outputs are written to
`/data/haziq/telept/my_scripts/data_visualization/rom_visualization_scripts/results/`.

## Commands for the Brett validation figures

/home/haziq/anaconda3/envs/mhr_new/bin/python /data/haziq/telept/my_scripts/data_visualization/rom_visualization_scripts/visualize_mhr_json_rom.py --json /home/haziq/Collab_AI/results/synthium/all_except_sa1b_70_30_brett_valfix/brett/s01_cam0_brett_adhesive_capsulitis.json --joint shoulder --movement flexion --time 10.076 --side right --output /data/haziq/telept/my_scripts/data_visualization/rom_visualization_scripts/results/json_valfix_adhesive_capsulitis_right_shoulder_flexion_t10p076.png

/home/haziq/anaconda3/envs/mhr_new/bin/python /data/haziq/telept/my_scripts/data_visualization/rom_visualization_scripts/visualize_mhr_json_rom.py --json /home/haziq/Collab_AI/results/synthium/all_except_sa1b_70_30_brett_valfix/brett/s01_cam0_brett_adhesive_capsulitis.json --joint shoulder --movement abduction --time 14.180 --side right --output /data/haziq/telept/my_scripts/data_visualization/rom_visualization_scripts/results/json_valfix_adhesive_capsulitis_right_shoulder_abduction_t14p180.png

/home/haziq/anaconda3/envs/mhr_new/bin/python /data/haziq/telept/my_scripts/data_visualization/rom_visualization_scripts/visualize_mhr_json_rom.py --json /home/haziq/Collab_AI/results/synthium/all_except_sa1b_70_30_brett_valfix/brett/s01_cam0_brett_adhesive_capsulitis.json --joint shoulder --movement external_rotation --time 18.918 --side right --output /data/haziq/telept/my_scripts/data_visualization/rom_visualization_scripts/results/json_valfix_adhesive_capsulitis_right_shoulder_external_rotation_t18p918.png

/home/haziq/anaconda3/envs/mhr_new/bin/python /data/haziq/telept/my_scripts/data_visualization/rom_visualization_scripts/visualize_mhr_json_rom.py --json /home/haziq/Collab_AI/results/synthium/all_except_sa1b_70_30_brett_valfix/brett/s01_cam0_brett_adhesive_capsulitis.json --joint shoulder --movement internal_rotation --time 20.085 --side right --output /data/haziq/telept/my_scripts/data_visualization/rom_visualization_scripts/results/json_valfix_adhesive_capsulitis_right_shoulder_internal_rotation_t20p085.png

/home/haziq/anaconda3/envs/mhr_new/bin/python /data/haziq/telept/my_scripts/data_visualization/rom_visualization_scripts/visualize_mhr_json_rom.py --json /home/haziq/Collab_AI/results/synthium/all_except_sa1b_70_30_brett_valfix/brett/s01_cam0_brett_hip_osteoarthritis.json --joint hip --movement flexion --time 9.175 --side right --output /data/haziq/telept/my_scripts/data_visualization/rom_visualization_scripts/results/json_valfix_hip_osteoarthritis_right_hip_flexion_t9p175.png

/home/haziq/anaconda3/envs/mhr_new/bin/python /data/haziq/telept/my_scripts/data_visualization/rom_visualization_scripts/visualize_mhr_json_rom.py --json /home/haziq/Collab_AI/results/synthium/all_except_sa1b_70_30_brett_valfix/brett/s01_cam0_brett_hip_osteoarthritis.json --joint hip --movement internal_rotation --time 14.446 --side right --output /data/haziq/telept/my_scripts/data_visualization/rom_visualization_scripts/results/json_valfix_hip_osteoarthritis_right_hip_internal_rotation_t14p446.png

/home/haziq/anaconda3/envs/mhr_new/bin/python /data/haziq/telept/my_scripts/data_visualization/rom_visualization_scripts/visualize_mhr_json_rom.py --json /home/haziq/Collab_AI/results/synthium/all_except_sa1b_70_30_brett_valfix/brett/s01_cam0_brett_hip_osteoarthritis.json --joint hip --movement external_rotation --time 18.283 --side right --output /data/haziq/telept/my_scripts/data_visualization/rom_visualization_scripts/results/json_valfix_hip_osteoarthritis_right_hip_external_rotation_t18p283.png

/home/haziq/anaconda3/envs/mhr_new/bin/python /data/haziq/telept/my_scripts/data_visualization/rom_visualization_scripts/visualize_mhr_json_rom.py --json /home/haziq/Collab_AI/results/synthium/all_except_sa1b_70_30_brett_valfix/brett/s01_cam0_brett_hip_osteoarthritis.json --joint hip --movement abduction --time 22.720 --side right --output /data/haziq/telept/my_scripts/data_visualization/rom_visualization_scripts/results/json_valfix_hip_osteoarthritis_right_hip_abduction_t22p720.png

/home/haziq/anaconda3/envs/mhr_new/bin/python /data/haziq/telept/my_scripts/data_visualization/rom_visualization_scripts/visualize_mhr_json_rom.py --json /home/haziq/Collab_AI/results/synthium/all_except_sa1b_70_30_brett_valfix/brett/s01_cam0_brett_hip_osteoarthritis.json --joint hip --movement extension --time 27.291 --side right --output /data/haziq/telept/my_scripts/data_visualization/rom_visualization_scripts/results/json_valfix_hip_osteoarthritis_right_hip_extension_t27p291.png

## Output layout

- The left panel is the clean source-video frame.
- The center panel is the body-aligned mesh with the ROM result and vectors.
- For planar movements, the right panel is the second body-aligned view.
- For axial rotations, the right panel is the full MHR body viewed along the
  humerus/thigh rotation axis.
