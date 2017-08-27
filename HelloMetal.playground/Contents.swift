let shaderCode =
    "#include <metal_stdlib>\n" +
        "using namespace metal;" +
        "kernel void doubler(const device float *inVector [[ buffer(0) ]]," +
        "device float *outVector [[ buffer(1) ]]," +
        "uint id [[ thread_position_in_grid ]]) {" +
        "    outVector[id] = 2*inVector[id];" +
"}"

import MetalKit
import PlaygroundSupport

let devices = MTLCopyAllDevices()
devices.count

devices[0].name
devices[0].areRasterOrderGroupsSupported
devices[0].isDepth24Stencil8PixelFormatSupported
devices[0].isLowPower
devices[0].maxThreadsPerThreadgroup.depth
devices[0].maxThreadgroupMemoryLength
devices[0].supportsFeatureSet(MTLFeatureSet.macOS_GPUFamily1_v3)
devices[0].supportsFeatureSet(MTLFeatureSet.macOS_ReadWriteTextureTier2)

devices[1].name
devices[1].areRasterOrderGroupsSupported
devices[1].isDepth24Stencil8PixelFormatSupported
devices[1].isLowPower
devices[1].maxThreadsPerThreadgroup.depth
devices[1].maxThreadgroupMemoryLength
devices[1].supportsFeatureSet(MTLFeatureSet.macOS_GPUFamily1_v3)
devices[1].supportsFeatureSet(MTLFeatureSet.macOS_ReadWriteTextureTier2)

// ------------------------------------------------------------------------------------------------------------------------
// Setup Metal
// ------------------------------------------------------------------------------------------------------------------------

// Get access to OSX dedicated GPU
let metalDevice:MTLDevice! = devices[1]

// Queue to handle an ordered list of command buffers
let metalCommandQueue:MTLCommandQueue! = metalDevice.makeCommandQueue()

// Buffer for storing encoded commands that are sent to GPU
let metalCommandBuffer:MTLCommandBuffer! = metalCommandQueue.makeCommandBuffer()


// ------------------------------------------------------------------------------------------------------------------------
// Setup Shader metal in pipeline
// ------------------------------------------------------------------------------------------------------------------------
let library = try! metalDevice.makeLibrary(source: shaderCode, options: nil)
let shader = library.makeFunction(name: "doubler")!
let computePipelineState:MTLComputePipelineState = try! metalDevice.makeComputePipelineState(function: shader)




// Create input and output vectors, and corresponding metal buffer
var inputVector = [Float](repeating: 0.0, count: 100)
for (index, _) in inputVector.enumerated() {
    inputVector[index] = Float(index)
}
let inputByteLength = inputVector.count * MemoryLayout.size(ofValue: Float())
let inputMetalBuffer = metalDevice.makeBuffer(bytes: &inputVector, length: inputByteLength, options: [])!

var outputVector = [Float](repeating: 0.0, count: 100)
let byteLength = outputVector.count * MemoryLayout.size(ofValue: Float())
let outputMetalBuffer = metalDevice.makeBuffer(bytes: &outputVector, length: byteLength, options: [])!

// Create Metal Compute Command Encoder and add input and output buffers to it
let metalComputeCommandEncoder:MTLComputeCommandEncoder! = metalCommandBuffer.makeComputeCommandEncoder()
metalComputeCommandEncoder!.setBuffer(inputMetalBuffer, offset: 0, index: 0)
metalComputeCommandEncoder!.setBuffer(outputMetalBuffer, offset: 0, index: 1)

// Set the shader function that Metal will use
metalComputeCommandEncoder!.setComputePipelineState(computePipelineState)

// Find max number of parallel GPU threads (threadExecutionWidth) in computePipelineState
let threadExecutionWidth = computePipelineState.threadExecutionWidth

// Set up thread groups on GPU
let threadsPerGroup = MTLSize(width:threadExecutionWidth, height: 1, depth: 1)
let numThreadgroups = MTLSize(width:(inputVector.count + threadExecutionWidth)/threadExecutionWidth, height: 1, depth: 1)
metalComputeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

// Finalize configuration
metalComputeCommandEncoder.endEncoding()

print("outputVector before job is running: \(outputVector)")

metalCommandBuffer.addCompletedHandler { cb in
    // Get output data from Metal/GPU into Swift
    let data = NSData(bytesNoCopy: outputMetalBuffer.contents(), length: outputVector.count * MemoryLayout.size(ofValue: Float()), freeWhenDone: false)
    data.getBytes(&outputVector, length: outputVector.count * MemoryLayout.size(ofValue: Float()))
    print("inputVector = \(inputVector)")
    print("outputVector = \(outputVector)")
}

// Start job
metalCommandBuffer.commit()
