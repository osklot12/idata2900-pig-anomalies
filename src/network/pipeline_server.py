import grpc
import pipeline_server_pb2
import pipeline_server_pb2_grpc
from concurrent import futures
import tensorflow as tf
import json
import asyncio

class PipelineServerServicer(pipeline_server_pb2_grpc.PipelineServerServicer):
    """gRPC server to receive processed data and stream it to training machines"""

    def __init__(self):
        self.memory_store = []  # 🔹 Stores processed frames + annotations in memory

    async def SendProcessedData(self, request_iterator, context):
        """Receives processed frames & annotations from the pipeline"""
        async for request in request_iterator:
            print(f"📡 Received frame {request.frame_name} from {request.video_name}")

            # Decode image
            image_tensor = tf.image.decode_jpeg(request.frame_data, channels=3)

            # Store in memory
            self.memory_store.append({
                "frame": image_tensor,
                "annotation": json.loads(request.annotation_json)
            })

        return pipeline_server_pb2.UploadResponse(message="✅ Processed data stored in memory")

    async def StreamProcessedBatches(self, request, context):
        """Streams processed data to training machines"""
        batch_size = request.batch_size

        while True:
            if len(self.memory_store) >= batch_size:
                batch = self.memory_store[:batch_size]
                self.memory_store = self.memory_store[batch_size:]  # Remove streamed batch

                frames = [tf.io.encode_jpeg(frame["frame"]).numpy() for frame in batch]
                annotations = [json.dumps(frame["annotation"]) for frame in batch]

                yield pipeline_server_pb2.ProcessedBatch(frames=frames, annotations=annotations)

            await asyncio.sleep(0.05)  # Prevent CPU overuse

def serve():
    server = grpc.aio.server()
    pipeline_server_pb2_grpc.add_PipelineServerServicer_to_server(PipelineServerServicer(), server)
    server.add_insecure_port("[::]:50051")
    print("🚀 Pipeline gRPC Server is running...")
    server.start()
    asyncio.get_event_loop().run_until_complete(server.wait_for_termination())

if __name__ == "__main__":
    serve()
