from ursina import *
from ursina.shaders import basic_lighting_shader
from utility.vectorUtil import ReferenceFrame, quaternion2euler
from rocket.primitives import *

app = Ursina()

# Right hand rule for triangles (just seems to be faces?) so normals face 'out' of the shape and we can see them
x = 2
y = 1.5
z = 0.2

rectPrismPrimitive = RectangularPrism(x, y, z, material=Aluminium)
primitiveFrame = ReferenceFrame(axis=np.array([0,0,1], float), ang=pi/2)

rectangular_prism_mesh = Mesh(vertices=[[-x,-y/2,-z/2],
                                    [0, -y/2, -z/2],
                                    [0, -y/2, z/2],
                                    [-x, -y/2, z/2],
                                    [-x, y/2, -z/2],
                                    [0, y/2, -z/2],
                                    [0, y/2, z/2],
                                    [-x, y/2, z/2]],

                            triangles=[[3,2,1,0],
                                    [0,1,5,4],
                                    [1,2,6,5],
                                    [2,3,7,6],
                                    [3,0,4,7],
                                    [4,5,6,7]],

                            normals=[[-1, -1, -1],
                                    [1, -1, -1],
                                    [1, -1, 1],
                                    [-1, -1, 1],
                                    [-1, 1, -1],
                                    [1, 1, -1],
                                    [1, 1, 1],
                                    [-1, 1, 1]])


rectPrism = Entity(model=rectangular_prism_mesh, shader=basic_lighting_shader, color=color.red)

roll, pitch, yaw = quaternion2euler(primitiveFrame.q)

print(f"roll: {roll*180/pi} deg, pitch: {pitch*180/pi} deg, yaw: {yaw * 180/pi} deg")

roll *= 180 / pi
pitch *= 180 / pi
yaw *= 180 / pi

rectPrism.rotation = Vec3(roll, pitch, yaw)

print(f"frame quaternion: {primitiveFrame.q}")

def showQuat(entity:Entity):
    print(f"entity quaternion: {entity.quaternion}")


showQuat(rectPrism)
EditorCamera()

app.run()