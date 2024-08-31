import Navbar from "@/components/Navbar";

export default function ThankYou(){
    return (
        <div>
            <Navbar />
            <div className="w-[100vw] h-[100vh] flex justify-center items-center font-staat text-primary text-5xl">
                Thank you for registering your data
            </div>
        </div>
    )
}