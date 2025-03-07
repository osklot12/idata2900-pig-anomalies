import grpc
import asyncio
import pipeline_server_pb2
import pipeline_server_pb2_grpc

class PipelineService(pipeline_server_pb2_grpc.PipelineServerServicer):
    async def SendProcessedData(self, request_iterator, context):
        async for request in request_iterator:
            print(f"📥 Received frame '{request.frame_name}' from video '{request.video_name}'")
            print(f"📑 Annotation JSON: {request.annotation_json[:100]}")  # Print first 100 chars
        return pipeline_server_pb2.UploadResponse(message="Frames uploaded successfully")

    async def StreamProcessedBatches(self, request, context):
        print(f"🟢 Worker received batch request of size {request.batch_size}")
        for i in range(request.batch_size):
            yield pipeline_server_pb2.ProcessedBatch(
                frames=[b"frame_bytes"],
                annotations=["{\"label\": \"object\"}"]
            )
            await asyncio.sleep(1)  # Simulate processing time

async def serve():
    server = grpc.aio.server()
    pipeline_server_pb2_grpc.add_PipelineServerServicer_to_server(PipelineService(), server)
    server.add_insecure_port("0.0.0.0:50051")
    await server.start()
    print("🚀 Worker Server Running... Waiting for processed frames.")
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())
