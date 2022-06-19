import './../styles/home.css';
import img from './../images/MedicalPrescription.png';

const Home = () => {
    return (
        <div>
            <div style={{ padding: "100px 0px", backgroundColor: "#7175da", textAlign: "center" }}>
                <h1 style={{ color: "white" }} className="main-title">Auto Annotation Platform</h1>
                <p className="desc">Tool to Annotate your digital documents on the go!</p>
            </div>
            <div style={{ paddingTop: '40px', textAlign: 'center' }}>
                <h2 className="sub-title">About tool</h2>
            </div>
            <div style={{ display: 'flex', justifyContent: 'center' }}>
                <p className="sub-desc">Annotator is an automatic annotation tool that supports the semantic annotation of digital documents.
                    Annotator can auto-annotate digital document of any language. We present you some snap shots of
                    auto-annotations for languages like Telugu, Hindi, Sanskrit, Arabic etc.
                </p>
            </div>
            <div style={{ paddingTop: "70px", display: 'flex', justifyContent: 'center', height: '900px' }}>
                <img src={img} alt="" />
            </div>
      </div >
    )
}

export default Home;