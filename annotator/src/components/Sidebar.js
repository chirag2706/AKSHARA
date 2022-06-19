import { useEffect, useState } from "react";
import "../styles/sidebar.css";

const Sidebar = (props) => {
    let ownedProjects = props.ownedProjects;
    let sharedProjects = props.sharedProjects;
    let project_id = props.project_id;
    let type = props.type;
    let path = props.path;

    const [projectName, setProjectName] = useState("")

    useEffect(() => {
        let project = ownedProjects.find((project) => project._id === project_id)
        if(project) {
            setProjectName(project.title)
        } else {
            project = sharedProjects.find((project) => project._id === project_id)
            if(project) {
                setProjectName(project.title)
            }
        }
    }, [])

    return (
        <div id="projectSideBar">
            <nav className="navbar navbar-light bg-light fixed-left">
                <div className="container-fluid">
                    <div>
                        <button className="navbar-toggler" type="button" data-bs-toggle="offcanvas" data-bs-target="#offcanvasNavbar" aria-controls="offcanvasNavbar" style={{display: "inline"}}>
                            <span className="navbar-toggler-icon"></span>
                        </button>
                        {type === "1" ? <h4 className="navbar-brand ms-4" style={{display: "inline"}}>{projectName}</h4> : (<></>)}
                    </div>
                    {type === "1" ? (
                        <button type="button" className="btn btn-primary" data-bs-toggle="modal" data-bs-target="#UploadDocumentModal">
                            Upload Document
                        </button>
                    ) : (
                        <></>
                    )}
                    <div className="offcanvas offcanvas-start" tabIndex="-1" id="offcanvasNavbar" aria-labelledby="offcanvasNavbarLabel">
                        <div className="offcanvas-header border-bottom">
                            {type === "1" ? (
                                <h5 className="offcanvas-title" id="offcanvasNavbarLabel">Projects</h5>
                            ) : (
                                <button type="button" className="btn btn-primary" data-bs-toggle="modal" data-bs-target="#CreateProjectModal">
                                    Create Project
                                </button>
                            )}

                            <button type="button" className="btn-close text-reset" data-bs-dismiss="offcanvas" aria-label="Close"></button>
                        </div>
                        <div className="offcanvas-body">
                            <ul className="navbar-nav justify-content-end flex-grow-1 pe-3">
                                {ownedProjects.length > 0 ? <li className="nav-item">Owned Projects</li> : <></>}
                                {ownedProjects.length > 0 ? (
                                    ownedProjects.map((project, index) => (
                                        <li className={"nav-item shadow-sm " + (project._id === project_id ? "" : "border")} key={index}>
                                            <a className={"nav-link " + (project._id === project_id ? "bg-dark" : "")} aria-current="page" href={path + '/' + project._id}>
                                                <div className={"project-title " + (project._id === project_id ? "text-white" : "")}>{project.title}</div>
                                                <div className={"project-creator-container " + (project._id === project_id ? "text-white" : "")}>
                                                    <span style={{ fontSize: "10px", paddingRight: "5px" }}>created by</span>
                                                    <span className="project-creator">{project.creator}</span>
                                                </div>
                                            </a>
                                        </li>
                                    ))
                                ) : (
                                    <></>
                                )}
                                {sharedProjects.length > 0 ? <li className="nav-item">Shared Projects</li> : <></>}
                                {sharedProjects.length > 0 ? (
                                    sharedProjects.map((project, index) => (
                                        <li className={"nav-item shadow-sm " + (project._id === project_id ? "" : "border")} key={index}>
                                            <a className={"nav-link " + (project._id === project_id ? "bg-dark" : "")} aria-current="page" href={path + '/' + project._id}>
                                                <div className={"project-title " + (project._id === project_id ? "text-white" : "")}>{project.title}</div>
                                                <div className={"project-creator-container " + (project._id === project_id ? "text-white" : "")}>
                                                    <span style={{ fontSize: "10px", paddingRight: "5px" }}>created by</span>
                                                    <span className="project-creator">{project.creator}</span>
                                                </div>
                                            </a>
                                        </li>
                                    ))
                                ) : (
                                    <></>
                                )}
                            </ul>
                        </div>
                    </div>
                </div>
            </nav>
        </div>
    );
};

export default Sidebar;
