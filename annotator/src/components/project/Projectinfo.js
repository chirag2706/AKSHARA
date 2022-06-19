import { connect } from "react-redux";

const Projectinfo = (props) => {
    let projectInfo = props.projectInfo;
    let handleLeaveProject = props.handleLeaveProject;

    return (
        <div className="container">
            <div className="p-4">
                <div className="d-grid">
                    <div className="p-2">
                        <div className="project-field-title">Title</div>
                        <div className="project-field-value">
                            {projectInfo["title"]}
                        </div>
                    </div>
                    <div className="p-2">
                        <div className="project-field-title">Description</div>
                        <div className={"project-field-value description" + (projectInfo["description"] === "" ? " blank-description" : "")}>
                            {projectInfo["description"]}
                        </div>
                    </div>
                    <div className="p-2">
                        <div className="project-field-title">Created by</div>
                        <div className="project-field-value">{projectInfo["creator"]}</div>
                    </div>
                </div>
                { props.auth.username ===  projectInfo["creator"] ? (<></>) : (
                    <button type="button" className="btn btn-danger" style={{ marginTop: "30px" }} onClick={() => {handleLeaveProject()}}>Leave Project</button>
                )}
            </div>
        </div>
    );
};

export default connect((state) => ({ auth: state.auth }))(Projectinfo);
