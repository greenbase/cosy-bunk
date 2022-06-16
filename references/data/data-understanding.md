# Raw data

The raw data consists of two parts. One is the position and orientation of joints in the human body as defined by the Azure Kincet system. Each joint is represented by its own coordinate system, which is positioned and rotated relative to the coordinate system defined by the depth camera of the Azure Kincet system.
The second part envelops the readings of a grid of force sensors on which the slats of a slatted frame rest.

For a single observation the data set contains positional and oriented data for 32 joints and 64 force measurements.

# Relation between raw data and sleeping positions

Not all features from the set of raw data is relevant to the task of determining the sleeping position of a person lying in a bed.  
First, the orientation of an individual joint is not required, because it does not have an relevant influence on the sleeping position. The tilting of a hand or the head for example does not alter the persons position on the bed in a meaningful way.  
Secondly, the rotation one limb may influences the position of another, e.g. the rotation of an elbow could have an influence on the position of the hand. Still this does not make it necessary to take the rotation into consideration since the position of the limb influenced by the rotation of another is sufficiently described by its own positional coordinates.  
In conclusion only the locations of the joints are needed to determine the sleeping position.

One can reduce the number of relevant features to the sleeping position &ndash; and therefore the number of output variables from the model &ndash; even further then only considering the position of the joints in plane. This is a reasonable simplification for the following reasons:

1. The Azure Kinect system is mounted central above the bed and facing straight downwards. The surface plane of the bed is therefore orthogonal to the z-axis of the global coordinate system. The z-coordinate for any point (or joint) in this plane is equal, thus carrying no relevant information about the position of the joints.
2. Consider a person laying a bed: All joints are located closely to a plane parallel to the surface of the bed, they all can approximately be considered to lie on this plane. This approximation is even applicable if the person is in a lateral position, because of reason No. 3.
3. Vertical differences in the position of joints are reflected by their positions in the plane. In a supine position for example the shoulder joints are located further apart from each other in the plane than if in a lateral position. The relative position of shoulder joints to each other can therefore serve as a sufficient indicator for a supine (or prone) or lateral sleeping position.
4. The orientation of a lateral sleeping position can be determined by only considering the joint positions in the plane, because their position relative to each other are characteristic to a certain orientation (left or right), e.g. the hand are not expected to be behind the back when in a lateral position.

For the reason above it is applicable to build a machine learning model to only predict the x- and y-coordinates of the joints.