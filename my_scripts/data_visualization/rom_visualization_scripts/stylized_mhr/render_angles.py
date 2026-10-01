"""Render the saved character without rebuilding or saving changes to the blend."""
import bpy, math, json, hashlib
from pathlib import Path
from mathutils import Vector
REPO_ROOT=Path(__file__).resolve().parents[3]
OUT=REPO_ROOT/'data'/'body_models'/'mhr'
PREVIEW_DIR=OUT/'previews'/'stylized_mhr_angles'
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
blend=OUT/'blender'/'stylized_mhr.blend'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
before={'scene':digest(blend)}
bpy.ops.wm.open_mainfile(filepath=str(blend))
scene=bpy.context.scene; camera=scene.camera
angles=[(0,'front_000'),(45,'azimuth_plus045'),(-45,'azimuth_minus045'),(135,'azimuth_plus135'),(-135,'azimuth_minus135'),(180,'rear_180')]
side_angles=[(90,'side_plus090'),(-90,'side_minus090')]
inventory=[]; supplementary=[]
for frame,pose in [(1,'neutral'),(40,'test')]:
    scene.frame_set(frame)
    for angle,label in angles+side_angles:
        a=math.radians(angle)
        camera.location=(4*math.sin(a),-4*math.cos(a),1.05)
        camera.rotation_euler=(Vector((0,0,.88))-camera.location).to_track_quat('-Z','Y').to_euler()
        filename=f'{pose}_{label}.png'
        scene.render.filepath=str(PREVIEW_DIR/filename)
        bpy.ops.render.render(write_still=True)
        collection=inventory if (angle,label) in angles else supplementary
        collection.append({'file':filename,'pose':pose,'frame':frame,'azimuth_degrees':angle,'camera_location_m':list(camera.location)})
after={'scene':digest(blend)}
assert before==after
(PREVIEW_DIR/'render_inventory.json').write_text(json.dumps({'source_blend':str(blend),'rotation':'azimuth about world Z; 0 looks from -Y, positive angles toward +X','camera_height_m':1.05,'orbit_radius_m':4,'target_m':[0,0,.88],'orthographic_scale':camera.data.ortho_scale,'resolution':[scene.render.resolution_x,scene.render.resolution_y],'preserved_file_sha256':after,'source_blend_and_deformation_validation_unchanged':True,'renders':inventory,'supplementary_side_renders':supplementary},indent=2))
