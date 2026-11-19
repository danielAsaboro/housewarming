// Download the current room as a self-contained MJCF bundle for desktop MuJoCo.
import { strToU8, zipSync } from 'fflate';

export function exportMjcf(id: string, sceneXml: string, robotFiles: readonly (readonly [string, Uint8Array])[]) {
  const files: Record<string, Uint8Array> = { [`${id}/scene.xml`]: strToU8(sceneXml) };
  for (const [f, bytes] of robotFiles) files[`${id}/${f}`] = bytes;
  files[`${id}/README.txt`] = strToU8(
    `Housewarming room "${id}"\n\npython -m mujoco.viewer --mjcf=scene.xml\n\nRobot: DEEP Robotics Lite3 (BSD-3, deep_robotics_model). Room: Aholo Spatial Understanding layout.\n`,
  );
  const blob = new Blob([zipSync(files)], { type: 'application/zip' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `housewarming-${id}-mjcf.zip`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}
